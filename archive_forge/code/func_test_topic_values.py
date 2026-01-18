import logging
import unittest
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import hdpmodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, common_texts
import numpy as np
def test_topic_values(self):
    """
        Check show topics method
        """
    results = self.model.show_topics()[0]
    expected_prob, expected_word = ('0.264', 'trees ')
    prob, word = results[1].split('+')[0].split('*')
    self.assertEqual(results[0], 0)
    self.assertEqual(prob, expected_prob)
    self.assertEqual(word, expected_word)
    return