import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
class TestBLEUFringeCases(unittest.TestCase):

    def test_case_where_n_is_bigger_than_hypothesis_length(self):
        references = ['John loves Mary ?'.split()]
        hypothesis = 'John loves Mary'.split()
        n = len(hypothesis) + 1
        weights = (1.0 / n,) * n
        self.assertAlmostEqual(sentence_bleu(references, hypothesis, weights), 0.0, places=4)
        try:
            self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
        except AttributeError:
            pass
        references = ['John loves Mary'.split()]
        hypothesis = 'John loves Mary'.split()
        self.assertAlmostEqual(sentence_bleu(references, hypothesis, weights), 0.0, places=4)

    def test_empty_hypothesis(self):
        references = ['The candidate has no alignment to any of the references'.split()]
        hypothesis = []
        assert sentence_bleu(references, hypothesis) == 0

    def test_length_one_hypothesis(self):
        references = ['The candidate has no alignment to any of the references'.split()]
        hypothesis = ['Foo']
        method4 = SmoothingFunction().method4
        try:
            sentence_bleu(references, hypothesis, smoothing_function=method4)
        except ValueError:
            pass

    def test_empty_references(self):
        references = [[]]
        hypothesis = 'John loves Mary'.split()
        assert sentence_bleu(references, hypothesis) == 0

    def test_empty_references_and_hypothesis(self):
        references = [[]]
        hypothesis = []
        assert sentence_bleu(references, hypothesis) == 0

    def test_reference_or_hypothesis_shorter_than_fourgrams(self):
        references = ['let it go'.split()]
        hypothesis = 'let go it'.split()
        self.assertAlmostEqual(sentence_bleu(references, hypothesis), 0.0, places=4)
        try:
            self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
        except AttributeError:
            pass