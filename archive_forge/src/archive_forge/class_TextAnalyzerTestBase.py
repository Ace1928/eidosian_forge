import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
class TextAnalyzerTestBase(unittest.TestCase):
    texts = [['this', 'is', 'a'], ['test', 'document'], ['this', 'test', 'document'], ['test', 'test', 'this']]
    token2id = {'this': 10, 'is': 15, 'a': 20, 'test': 21, 'document': 17}
    dictionary = Dictionary(texts)
    dictionary.token2id = token2id
    dictionary.id2token = {v: k for k, v in token2id.items()}
    top_ids = set(token2id.values())
    texts2 = common_texts + [['user', 'user']]
    dictionary2 = Dictionary(texts2)
    dictionary2.id2token = {v: k for k, v in dictionary2.token2id.items()}
    top_ids2 = set(dictionary2.token2id.values())
    accumulator_cls = None

    def init_accumulator(self):
        return self.accumulator_cls(self.top_ids, self.dictionary)

    def init_accumulator2(self):
        return self.accumulator_cls(self.top_ids2, self.dictionary2)

    def test_occurrence_counting(self):
        accumulator = self.init_accumulator().accumulate(self.texts, 3)
        self.assertEqual(3, accumulator.get_occurrences('this'))
        self.assertEqual(1, accumulator.get_occurrences('is'))
        self.assertEqual(1, accumulator.get_occurrences('a'))
        self.assertEqual(2, accumulator.get_co_occurrences('test', 'document'))
        self.assertEqual(2, accumulator.get_co_occurrences('test', 'this'))
        self.assertEqual(1, accumulator.get_co_occurrences('is', 'a'))

    def test_occurrence_counting2(self):
        accumulator = self.init_accumulator2().accumulate(self.texts2, 110)
        self.assertEqual(2, accumulator.get_occurrences('human'))
        self.assertEqual(4, accumulator.get_occurrences('user'))
        self.assertEqual(3, accumulator.get_occurrences('graph'))
        self.assertEqual(3, accumulator.get_occurrences('trees'))
        cases = [(1, ('human', 'interface')), (2, ('system', 'user')), (2, ('graph', 'minors')), (2, ('graph', 'trees')), (4, ('user', 'user')), (3, ('graph', 'graph')), (0, ('time', 'eps'))]
        for expected_count, (word1, word2) in cases:
            self.assertEqual(expected_count, accumulator.get_co_occurrences(word1, word2))
            self.assertEqual(expected_count, accumulator.get_co_occurrences(word2, word1))
            word_id1 = self.dictionary2.token2id[word1]
            word_id2 = self.dictionary2.token2id[word2]
            self.assertEqual(expected_count, accumulator.get_co_occurrences(word_id1, word_id2))
            self.assertEqual(expected_count, accumulator.get_co_occurrences(word_id2, word_id1))

    def test_occurences_for_irrelevant_words(self):
        accumulator = self.init_accumulator().accumulate(self.texts, 2)
        with self.assertRaises(KeyError):
            accumulator.get_occurrences('irrelevant')
        with self.assertRaises(KeyError):
            accumulator.get_co_occurrences('test', 'irrelevant')