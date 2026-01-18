import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_push_does_not_add_hypothesis_that_falls_below_beam_threshold(self):
    stack = _Stack(3, 0.5)
    poor_hypothesis = _Hypothesis(0.01)
    stack.push(_Hypothesis(0.9))
    stack.push(poor_hypothesis)
    self.assertFalse(poor_hypothesis in stack)