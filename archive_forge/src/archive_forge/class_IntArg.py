import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
class IntArg:

    def __init__(self, a=-1000, b=1000):
        self.a = a
        self.b = b

    def values(self, n):
        v1 = Arg(self.a, self.b).values(max(1 + n // 2, n - 5)).astype(int)
        v2 = np.arange(-5, 5)
        v = np.unique(np.r_[v1, v2])
        v = v[(v >= self.a) & (v < self.b)]
        return v