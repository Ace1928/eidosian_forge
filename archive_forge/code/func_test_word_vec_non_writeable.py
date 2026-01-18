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
def test_word_vec_non_writeable(self):
    model = keyedvectors.KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'))
    vector = model['says']
    with self.assertRaises(ValueError):
        vector *= 0