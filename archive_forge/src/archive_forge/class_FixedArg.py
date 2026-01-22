import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
class FixedArg:

    def __init__(self, values):
        self._values = np.asarray(values)

    def values(self, n):
        return self._values