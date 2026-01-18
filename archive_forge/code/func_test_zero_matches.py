import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_zero_matches(self):
    references = ['The candidate has no alignment to any of the references'.split()]
    hypothesis = 'John loves Mary'.split()
    for n in range(1, len(hypothesis)):
        weights = (1.0 / n,) * n
        assert sentence_bleu(references, hypothesis, weights) == 0