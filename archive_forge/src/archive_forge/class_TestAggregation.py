import logging
import unittest
from gensim.topic_coherence import aggregation
class TestAggregation(unittest.TestCase):

    def setUp(self):
        self.confirmed_measures = [1.1, 2.2, 3.3, 4.4]

    def test_arithmetic_mean(self):
        """Test arithmetic_mean()"""
        obtained = aggregation.arithmetic_mean(self.confirmed_measures)
        expected = 2.75
        self.assertEqual(obtained, expected)