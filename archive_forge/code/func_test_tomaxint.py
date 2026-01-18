import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_tomaxint(self):
    random.seed(self.seed)
    rs = random.RandomState(self.seed)
    actual = rs.tomaxint(size=(3, 2))
    if np.iinfo(int).max == 2147483647:
        desired = np.array([[1328851649, 731237375], [1270502067, 320041495], [1908433478, 499156889]], dtype=np.int64)
    else:
        desired = np.array([[5707374374421908479, 5456764827585442327], [8196659375100692377, 8224063923314595285], [4220315081820346526, 7177518203184491332]], dtype=np.int64)
    assert_equal(actual, desired)
    rs.seed(self.seed)
    actual = rs.tomaxint()
    assert_equal(actual, desired[0, 0])