import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_valid_phrases(self):
    hypothesis = _Hypothesis()
    hypothesis.untranslated_spans = lambda _: [(0, 2), (3, 6)]
    all_phrases_from = [[1, 4], [2], [], [5], [5, 6, 7], [], [7]]
    phrase_spans = StackDecoder.valid_phrases(all_phrases_from, hypothesis)
    self.assertEqual(phrase_spans, [(0, 1), (1, 2), (3, 5), (4, 5), (4, 6)])