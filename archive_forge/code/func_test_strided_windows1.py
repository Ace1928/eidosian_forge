import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_strided_windows1(self):
    out = utils.strided_windows(range(5), 2)
    expected = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    self._assert_arrays_equal(expected, out)