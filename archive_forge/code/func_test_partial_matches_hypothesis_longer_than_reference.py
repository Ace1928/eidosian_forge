import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_partial_matches_hypothesis_longer_than_reference(self):
    references = ['John loves Mary'.split()]
    hypothesis = 'John loves Mary who loves Mike'.split()
    self.assertAlmostEqual(sentence_bleu(references, hypothesis), 0.0, places=4)
    try:
        self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
    except AttributeError:
        pass