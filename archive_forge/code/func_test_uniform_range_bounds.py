import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_uniform_range_bounds(self):
    fmin = np.finfo('float').min
    fmax = np.finfo('float').max
    func = random.uniform
    assert_raises(OverflowError, func, -np.inf, 0)
    assert_raises(OverflowError, func, 0, np.inf)
    assert_raises(OverflowError, func, fmin, fmax)
    assert_raises(OverflowError, func, [-np.inf], [0])
    assert_raises(OverflowError, func, [0], [np.inf])
    random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e+17)