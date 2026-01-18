import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_strided_windows_window_size_exceeds_size(self):
    input_arr = np.array(['this', 'is', 'test'], dtype='object')
    out = utils.strided_windows(input_arr, 4)
    expected = np.ndarray((0, 0))
    self._assert_arrays_equal(expected, out)