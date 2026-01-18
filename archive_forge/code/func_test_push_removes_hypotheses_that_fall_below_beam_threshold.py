import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_push_removes_hypotheses_that_fall_below_beam_threshold(self):
    stack = _Stack(3, 0.5)
    poor_hypothesis = _Hypothesis(0.01)
    worse_hypothesis = _Hypothesis(0.009)
    stack.push(poor_hypothesis)
    stack.push(worse_hypothesis)
    stack.push(_Hypothesis(0.9))
    self.assertFalse(poor_hypothesis in stack)
    self.assertFalse(worse_hypothesis in stack)