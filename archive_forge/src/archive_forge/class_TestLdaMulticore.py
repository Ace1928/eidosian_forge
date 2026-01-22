import logging
import numbers
import os
import unittest
import copy
import numpy as np
from numpy.testing import assert_allclose
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils, utils
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_texts
class TestLdaMulticore(TestLdaModel):

    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamulticore.LdaMulticore
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def test_alpha_auto(self):
        self.assertRaises(RuntimeError, self.class_, alpha='auto')