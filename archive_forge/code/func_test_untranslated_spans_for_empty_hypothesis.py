import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_untranslated_spans_for_empty_hypothesis(self):
    hypothesis = _Hypothesis()
    untranslated_spans = hypothesis.untranslated_spans(10)
    self.assertEqual(untranslated_spans, [(0, 10)])