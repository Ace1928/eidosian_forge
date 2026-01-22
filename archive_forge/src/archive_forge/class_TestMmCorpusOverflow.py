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
class TestMmCorpusOverflow(CorpusTestCase):
    """
    Test to make sure cython mmreader doesn't overflow on large number of docs or terms

    """

    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_overflow.mm'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        pass

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 44270060)
        self.assertEqual(self.corpus.num_terms, 500)
        self.assertEqual(self.corpus.num_nnz, 22134988630)
        it = iter(self.corpus)
        self.assertEqual(next(it)[:3], [(0, 0.3913027376444812), (1, -0.07658791716226626), (2, -0.020870794080588395)])
        self.assertEqual(next(it), [])
        self.assertEqual(next(it), [])
        count = 0
        for doc in self.corpus:
            for term in doc:
                count += 1
        self.assertEqual(count, 12)
        self.assertRaises(RuntimeError, lambda: self.corpus[3])