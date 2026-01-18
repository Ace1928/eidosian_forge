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
def test_sync_state(self):
    model2 = self.class_(corpus=self.corpus, id2word=dictionary, num_topics=2, passes=1)
    model2.state = copy.deepcopy(self.model.state)
    model2.sync_state()
    assert_allclose(self.model.get_term_topics(2), model2.get_term_topics(2), rtol=1e-05)
    assert_allclose(self.model.get_topics(), model2.get_topics(), rtol=1e-05)
    self.model.random_state = np.random.RandomState(0)
    model2.random_state = np.random.RandomState(0)
    self.model.passes = 1
    model2.passes = 1
    self.model.update(self.corpus)
    model2.update(self.corpus)
    assert_allclose(self.model.get_term_topics(2), model2.get_term_topics(2), rtol=1e-05)
    assert_allclose(self.model.get_topics(), model2.get_topics(), rtol=1e-05)