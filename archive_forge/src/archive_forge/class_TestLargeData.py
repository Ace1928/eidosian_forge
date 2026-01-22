import logging
import unittest
import os
import numpy as np
import gensim
from gensim.test.utils import get_tmpfile
class TestLargeData(unittest.TestCase):
    """Try common operations, using large models. You'll need ~8GB RAM to run these tests"""

    def test_word2vec(self):
        corpus = BigCorpus(words_only=True, num_docs=100000, num_terms=3000000, doc_len=200)
        tmpf = get_tmpfile('gensim_big.tst')
        model = gensim.models.Word2Vec(corpus, vector_size=300, workers=4)
        model.save(tmpf, ignore=['syn1'])
        del model
        gensim.models.Word2Vec.load(tmpf)

    def test_lsi_model(self):
        corpus = BigCorpus(num_docs=50000)
        tmpf = get_tmpfile('gensim_big.tst')
        model = gensim.models.LsiModel(corpus, num_topics=500, id2word=corpus.dictionary)
        model.save(tmpf)
        del model
        gensim.models.LsiModel.load(tmpf)

    def test_lda_model(self):
        corpus = BigCorpus(num_docs=5000)
        tmpf = get_tmpfile('gensim_big.tst')
        model = gensim.models.LdaModel(corpus, num_topics=500, id2word=corpus.dictionary)
        model.save(tmpf)
        del model
        gensim.models.LdaModel.load(tmpf)