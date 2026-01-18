import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_reference_or_hypothesis_shorter_than_fourgrams(self):
    references = ['let it go'.split()]
    hypothesis = 'let go it'.split()
    self.assertAlmostEqual(sentence_bleu(references, hypothesis), 0.0, places=4)
    try:
        self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
    except AttributeError:
        pass