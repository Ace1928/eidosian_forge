import logging
import unittest
import numpy as np
from gensim.topic_coherence import segmentation
from numpy import array
def test_s_one_pre(self):
    """Test s_one_pre segmentation."""
    actual = segmentation.s_one_pre(self.topics)
    expected = [[(4, 9), (6, 9), (6, 4)], [(10, 9), (7, 9), (7, 10)], [(2, 5), (7, 5), (7, 2)]]
    self.assertTrue(np.allclose(actual, expected))