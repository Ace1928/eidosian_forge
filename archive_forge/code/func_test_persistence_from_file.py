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
def test_persistence_from_file(self):
    """Test storing/loading the entire model trained with corpus_file argument."""
    with temporary_file(get_tmpfile('gensim_word2vec.tst')) as corpus_file:
        utils.save_as_line_sentence(sentences, corpus_file)
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(corpus_file=corpus_file, min_count=1)
        model.save(tmpf)
        self.models_equal(model, word2vec.Word2Vec.load(tmpf))
        wv = model.wv
        wv.save(tmpf)
        loaded_wv = keyedvectors.KeyedVectors.load(tmpf)
        self.assertTrue(np.allclose(wv.vectors, loaded_wv.vectors))
        self.assertEqual(len(wv), len(loaded_wv))