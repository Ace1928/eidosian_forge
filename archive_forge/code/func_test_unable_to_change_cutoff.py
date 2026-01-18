import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_unable_to_change_cutoff(self):
    with self.assertRaises(AttributeError):
        self.vocab.cutoff = 3