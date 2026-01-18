import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_translation_so_far_for_empty_hypothesis(self):
    hypothesis = _Hypothesis()
    translation = hypothesis.translation_so_far()
    self.assertEqual(translation, [])