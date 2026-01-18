import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import tfidfmodel
from gensim.test.utils import datapath, get_tmpfile, common_dictionary, common_corpus
from gensim.corpora import Dictionary
def test_wlocal_wglobal(self):

    def wlocal(tf):
        assert isinstance(tf, np.ndarray)
        return iter(tf + 1)

    def wglobal(df, total_docs):
        return 1
    docs = [corpus[1], corpus[2]]
    model = tfidfmodel.TfidfModel(corpus, wlocal=wlocal, wglobal=wglobal, normalize=False)
    transformed_docs = [model[docs[0]], model[docs[1]]]
    expected_docs = [[(termid, weight + 1) for termid, weight in docs[0]], [(termid, weight + 1) for termid, weight in docs[1]]]
    self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
    self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))