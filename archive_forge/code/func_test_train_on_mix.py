import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_train_on_mix(self):
    mixed_sent = [('a', 'b'), ('c', 'd'), ('e', 'f', 'g'), ('h',)]
    counter = NgramCounter([mixed_sent])
    unigrams = ['h']
    bigram_contexts = [('a',), ('c',)]
    trigram_contexts = [('e', 'f')]
    self.case.assertCountEqual(unigrams, counter[1].keys())
    self.case.assertCountEqual(bigram_contexts, counter[2].keys())
    self.case.assertCountEqual(trigram_contexts, counter[3].keys())