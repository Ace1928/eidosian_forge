from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def test_persistence_word2vec_format(self):
    """Test storing the entire model in word2vec format."""
    model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
    test_doc_word = get_tmpfile('gensim_doc2vec.dw')
    model.save_word2vec_format(test_doc_word, doctag_vec=True, word_vec=True, binary=False)
    binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_doc_word, binary=False)
    self.assertEqual(len(model.wv) + len(model.dv), len(binary_model_dv))
    test_doc = get_tmpfile('gensim_doc2vec.d')
    model.save_word2vec_format(test_doc, doctag_vec=True, word_vec=False, binary=True)
    binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_doc, binary=True)
    self.assertEqual(len(model.dv), len(binary_model_dv))
    test_word = get_tmpfile('gensim_doc2vec.w')
    model.save_word2vec_format(test_word, doctag_vec=False, word_vec=True, binary=True)
    binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_word, binary=True)
    self.assertEqual(len(model.wv), len(binary_model_dv))