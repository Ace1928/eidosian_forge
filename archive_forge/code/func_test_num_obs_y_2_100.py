import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_num_obs_y_2_100(self):
    a = set()
    for n in range(2, 16):
        a.add(n * (n - 1) / 2)
    for i in range(5, 105):
        if i not in a:
            with pytest.raises(ValueError):
                self.bad_y(i)