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
def test_random_state_backward_compatibility(self):
    pre_0_13_2_fname = datapath('pre_0_13_2_model')
    model_pre_0_13_2 = self.class_.load(pre_0_13_2_fname)
    model_topics = model_pre_0_13_2.print_topics(num_topics=2, num_words=3)
    for i in model_topics:
        self.assertTrue(isinstance(i[0], int))
        self.assertTrue(isinstance(i[1], str))
    post_0_13_2_fname = get_tmpfile('gensim_models_lda_post_0_13_2_model.tst')
    model_pre_0_13_2.save(post_0_13_2_fname)
    model_post_0_13_2 = self.class_.load(post_0_13_2_fname)
    model_topics_new = model_post_0_13_2.print_topics(num_topics=2, num_words=3)
    for i in model_topics_new:
        self.assertTrue(isinstance(i[0], int))
        self.assertTrue(isinstance(i[1], str))