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
def test_sokalmichener():
    p = [True, True, False]
    q = [True, False, True]
    x = [int(b) for b in p]
    y = [int(b) for b in q]
    dist1 = sokalmichener(p, q)
    dist2 = sokalmichener(x, y)
    assert_equal(dist1, dist2)