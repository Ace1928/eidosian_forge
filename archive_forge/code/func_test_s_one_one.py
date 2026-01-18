import logging
import unittest
import numpy as np
from gensim.topic_coherence import segmentation
from numpy import array
def test_s_one_one(self):
    """Test s_one_one segmentation."""
    actual = segmentation.s_one_one(self.topics)
    expected = [[(9, 4), (9, 6), (4, 9), (4, 6), (6, 9), (6, 4)], [(9, 10), (9, 7), (10, 9), (10, 7), (7, 9), (7, 10)], [(5, 2), (5, 7), (2, 5), (2, 7), (7, 5), (7, 2)]]
    self.assertTrue(np.allclose(actual, expected))