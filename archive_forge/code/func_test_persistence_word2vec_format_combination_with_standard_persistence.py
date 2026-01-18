import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_persistence_word2vec_format_combination_with_standard_persistence(self):
    """Test storing/loading the entire model and vocabulary in word2vec format chained with
         saving and loading via `save` and `load` methods`.
         It was possible prior to 1.0.0 release, now raises Exception"""
    tmpf = get_tmpfile('gensim_word2vec.tst')
    model = word2vec.Word2Vec(sentences, min_count=1)
    testvocab = get_tmpfile('gensim_word2vec.vocab')
    model.wv.save_word2vec_format(tmpf, testvocab, binary=True)
    binary_model_with_vocab_kv = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, testvocab, binary=True)
    binary_model_with_vocab_kv.save(tmpf)
    self.assertRaises(AttributeError, word2vec.Word2Vec.load, tmpf)