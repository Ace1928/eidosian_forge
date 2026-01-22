import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
class TestIsCorpus(unittest.TestCase):

    def test_None(self):
        result = utils.is_corpus(None)
        expected = (False, None)
        self.assertEqual(expected, result)

    def test_simple_lists_of_tuples(self):
        potentialCorpus = [[(0, 4.0)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)
        potentialCorpus = [[(0, 4.0), (1, 2.0)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)
        potentialCorpus = [[(0, 4.0), (1, 2.0), (2, 5.0), (3, 8.0)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)
        potentialCorpus = [[(0, 4.0)], [(1, 2.0)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)
        potentialCorpus = [[(0, 4.0)], [(1, 2.0)], [(2, 5.0)], [(3, 8.0)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

    def test_int_tuples(self):
        potentialCorpus = [[(0, 4)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

    def test_invalid_formats(self):
        potentials = list()
        potentials.append(['human'])
        potentials.append('human')
        potentials.append(['human', 'star'])
        potentials.append([1, 2, 3, 4, 5, 5])
        potentials.append([[(0, 'string')]])
        for noCorpus in potentials:
            result = utils.is_corpus(noCorpus)
            expected = (False, noCorpus)
            self.assertEqual(expected, result)