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
class TestSvmLightCorpus(CorpusTestCase):

    def setUp(self):
        self.corpus_class = svmlightcorpus.SvmLightCorpus
        self.file_extension = '.svmlight'

    def test_serialization(self):
        path = get_tmpfile('svml.corpus')
        labels = [1] * len(common_corpus)
        second_corpus = [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)]
        self.corpus_class.serialize(path, common_corpus, labels=labels)
        serialized_corpus = self.corpus_class(path)
        self.assertEqual(serialized_corpus[1], second_corpus)
        self.corpus_class.serialize(path, common_corpus, labels=np.array(labels))
        serialized_corpus = self.corpus_class(path)
        self.assertEqual(serialized_corpus[1], second_corpus)