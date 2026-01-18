import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_unigram_counts_completely_unseen_words(self):
    assert self.bigram_counter['z'] == 0