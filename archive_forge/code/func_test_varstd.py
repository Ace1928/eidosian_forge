import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_varstd(self):
    x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
    assert_almost_equal(mX.var(axis=None), mX.compressed().var())
    assert_almost_equal(mX.std(axis=None), mX.compressed().std())
    assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
    assert_equal(mX.var().shape, X.var().shape)
    mXvar0, mXvar1 = (mX.var(axis=0), mX.var(axis=1))
    assert_almost_equal(mX.var(axis=None, ddof=2), mX.compressed().var(ddof=2))
    assert_almost_equal(mX.std(axis=None, ddof=2), mX.compressed().std(ddof=2))
    for k in range(6):
        assert_almost_equal(mXvar1[k], mX[k].compressed().var())
        assert_almost_equal(mXvar0[k], mX[:, k].compressed().var())
        assert_almost_equal(np.sqrt(mXvar0[k]), mX[:, k].compressed().std())