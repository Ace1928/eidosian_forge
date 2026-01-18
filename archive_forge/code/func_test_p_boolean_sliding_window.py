import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.hashdictionary import HashDictionary
from gensim.topic_coherence import probability_estimation
def test_p_boolean_sliding_window(self):
    """Test p_boolean_sliding_window()"""
    accumulator = probability_estimation.p_boolean_sliding_window(self.texts, self.segmented_topics, self.dictionary, 2)
    self.assertEqual(1, accumulator[self.computer_id])
    self.assertEqual(3, accumulator[self.user_id])
    self.assertEqual(1, accumulator[self.graph_id])
    self.assertEqual(4, accumulator[self.system_id])