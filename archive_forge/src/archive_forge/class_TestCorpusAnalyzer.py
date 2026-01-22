import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
class TestCorpusAnalyzer(unittest.TestCase):

    def setUp(self):
        self.dictionary = BaseTestCases.TextAnalyzerTestBase.dictionary
        self.top_ids = BaseTestCases.TextAnalyzerTestBase.top_ids
        self.corpus = [self.dictionary.doc2bow(doc) for doc in BaseTestCases.TextAnalyzerTestBase.texts]

    def test_index_accumulation(self):
        accumulator = CorpusAccumulator(self.top_ids).accumulate(self.corpus)
        inverted_index = accumulator.index_to_dict()
        expected = {10: {0, 2, 3}, 15: {0}, 20: {0}, 21: {1, 2, 3}, 17: {1, 2}}
        self.assertDictEqual(expected, inverted_index)
        self.assertEqual(3, accumulator.get_occurrences(10))
        self.assertEqual(2, accumulator.get_occurrences(17))
        self.assertEqual(2, accumulator.get_co_occurrences(10, 21))
        self.assertEqual(1, accumulator.get_co_occurrences(10, 17))