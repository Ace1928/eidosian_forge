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
class TestMaskedArrayMathMethodsComplex:

    def setup_method(self):
        x = np.array([8.375j, 7.545j, 8.828j, 8.5j, 1.757j, 5.928, 8.43, 7.78, 9.865, 5.878, 8.979, 4.732, 3.012, 6.022, 5.095, 3.116, 5.238, 3.957, 6.04, 9.63, 7.712, 3.382, 4.489, 6.479j, 7.189j, 9.645, 5.395, 4.961, 9.894, 2.893, 7.357, 9.828, 6.272, 3.758, 6.693, 0.993j])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))
        m2 = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1])
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

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