import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_translated_positions(self):
    translated_positions = self.hypothesis_chain.translated_positions()
    translated_positions.sort()
    self.assertEqual(translated_positions, [1, 3, 4, 5, 6])