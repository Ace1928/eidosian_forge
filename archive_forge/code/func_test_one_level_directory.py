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
def test_one_level_directory(self):
    dirpath = self.write_one_level()
    corpus = textcorpus.TextDirectoryCorpus(dirpath)
    self.assertEqual(len(corpus), 2)
    docs = list(corpus)
    self.assertEqual(len(docs), 2)