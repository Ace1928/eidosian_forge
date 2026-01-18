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
def test_term_topics(self):
    model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
    result = model.get_term_topics(2)
    for topic_no, probability in result:
        self.assertTrue(isinstance(topic_no, int))
        self.assertTrue(np.issubdtype(probability, np.floating))
    result = model.get_term_topics(str(model.id2word[2]))
    for topic_no, probability in result:
        self.assertTrue(isinstance(topic_no, int))
        self.assertTrue(np.issubdtype(probability, np.floating))