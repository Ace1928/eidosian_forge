import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
class TestTrimVocabByFreq(unittest.TestCase):

    def test_trim_vocab(self):
        d = {'word1': 5, 'word2': 1, 'word3': 2}
        expected_dict = {'word1': 5, 'word3': 2}
        utils.trim_vocab_by_freq(d, topk=2)
        self.assertEqual(d, expected_dict)
        d = {'word1': 5, 'word2': 2, 'word3': 2, 'word4': 1}
        expected_dict = {'word1': 5, 'word2': 2, 'word3': 2}
        utils.trim_vocab_by_freq(d, topk=2)
        self.assertEqual(d, expected_dict)