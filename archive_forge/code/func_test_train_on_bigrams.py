import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_train_on_bigrams(self):
    bigram_sent = [('a', 'b'), ('c', 'd')]
    counter = NgramCounter([bigram_sent])
    assert not bool(counter[3])