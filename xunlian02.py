from gensim.models.word2vec import Word2Vec, LineSentence
import multiprocessing

print('开始训练')
model = Word2Vec(LineSentence('D:/xinjian/course/fenci01/zh_wiki_04.txt'), size=300, workers=multiprocessing.cpu_count())
print('结束')
model.init_sims(replace=True)
model.save('D:/xinjian/course/word/wiki_corpus_04_predict.model')

