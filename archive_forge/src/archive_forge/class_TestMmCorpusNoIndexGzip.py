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
class TestMmCorpusNoIndexGzip(CorpusTestCase):

    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_no_index.mm.gz'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        pass

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 9)
        self.assertEqual(self.corpus.num_terms, 12)
        self.assertEqual(self.corpus.num_nnz, 28)
        it = iter(self.corpus)
        self.assertEqual(next(it), [(0, 1.0), (1, 1.0), (2, 1.0)])
        self.assertEqual(next(it), [])
        self.assertEqual(next(it), [(2, 0.42371910849), (5, 0.6625174), (7, 1.0), (8, 1.0)])
        self.assertRaises(RuntimeError, lambda: self.corpus[3])