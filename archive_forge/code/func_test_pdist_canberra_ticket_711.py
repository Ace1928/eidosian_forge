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
def test_pdist_canberra_ticket_711(self):
    eps = 1e-08
    pdist_y = wpdist_no_const(([3.3], [3.4]), 'canberra')
    right_y = 0.01492537
    assert_allclose(pdist_y, right_y, atol=eps, verbose=verbose > 2)