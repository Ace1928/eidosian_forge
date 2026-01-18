import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_noncommutative_reduce_accumulate(self):
    tosubtract = np.arange(5)
    todivide = np.array([2.0, 0.5, 0.25])
    assert_equal(np.subtract.reduce(tosubtract), -10)
    assert_equal(np.divide.reduce(todivide), 16.0)
    assert_array_equal(np.subtract.accumulate(tosubtract), np.array([0, -1, -3, -6, -10]))
    assert_array_equal(np.divide.accumulate(todivide), np.array([2.0, 4.0, 16.0]))