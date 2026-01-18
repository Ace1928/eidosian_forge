import unittest
import logging
import numpy as np  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
def test_topic_word(self):
    topics = self.ldaseq.print_topics(0)
    expected_topic_word = [('skills', 0.036)]
    self.assertEqual(topics[0][0][0], expected_topic_word[0][0])
    self.assertAlmostEqual(topics[0][0][1], expected_topic_word[0][1], delta=0.0012)