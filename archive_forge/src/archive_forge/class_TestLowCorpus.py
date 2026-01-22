from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
class TestLowCorpus(CorpusTestCase):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]
    CORPUS_LINE = 'mom  wash  window window was washed'

    def setUp(self):
        self.corpus_class = lowcorpus.LowCorpus
        self.file_extension = '.low'

    def test_line2doc(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        id2word = {1: 'mom', 2: 'window'}
        corpus = self.corpus_class(fname, id2word=id2word)
        corpus.use_wordids = False
        self.assertEqual(sorted(corpus.line2doc(self.CORPUS_LINE)), [('mom', 1), ('was', 1), ('wash', 1), ('washed', 1), ('window', 2)])
        corpus.use_wordids = True
        self.assertEqual(sorted(corpus.line2doc(self.CORPUS_LINE)), [(1, 1), (2, 2)])