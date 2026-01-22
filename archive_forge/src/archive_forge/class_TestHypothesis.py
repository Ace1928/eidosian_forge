import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
class TestHypothesis(unittest.TestCase):

    def setUp(self):
        root = _Hypothesis()
        child = _Hypothesis(raw_score=0.5, src_phrase_span=(3, 7), trg_phrase=('hello', 'world'), previous=root)
        grandchild = _Hypothesis(raw_score=0.4, src_phrase_span=(1, 2), trg_phrase=('and', 'goodbye'), previous=child)
        self.hypothesis_chain = grandchild

    def test_translation_so_far(self):
        translation = self.hypothesis_chain.translation_so_far()
        self.assertEqual(translation, ['hello', 'world', 'and', 'goodbye'])

    def test_translation_so_far_for_empty_hypothesis(self):
        hypothesis = _Hypothesis()
        translation = hypothesis.translation_so_far()
        self.assertEqual(translation, [])

    def test_total_translated_words(self):
        total_translated_words = self.hypothesis_chain.total_translated_words()
        self.assertEqual(total_translated_words, 5)

    def test_translated_positions(self):
        translated_positions = self.hypothesis_chain.translated_positions()
        translated_positions.sort()
        self.assertEqual(translated_positions, [1, 3, 4, 5, 6])

    def test_untranslated_spans(self):
        untranslated_spans = self.hypothesis_chain.untranslated_spans(10)
        self.assertEqual(untranslated_spans, [(0, 1), (2, 3), (7, 10)])

    def test_untranslated_spans_for_empty_hypothesis(self):
        hypothesis = _Hypothesis()
        untranslated_spans = hypothesis.untranslated_spans(10)
        self.assertEqual(untranslated_spans, [(0, 10)])