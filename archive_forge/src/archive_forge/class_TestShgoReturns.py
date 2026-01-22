import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
class TestShgoReturns:

    def test_1_nfev_simplicial(self):
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        def fun(x):
            fun.nfev += 1
            return rosen(x)
        fun.nfev = 0
        result = shgo(fun, bounds)
        numpy.testing.assert_equal(fun.nfev, result.nfev)

    def test_1_nfev_sobol(self):
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        def fun(x):
            fun.nfev += 1
            return rosen(x)
        fun.nfev = 0
        result = shgo(fun, bounds, sampling_method='sobol')
        numpy.testing.assert_equal(fun.nfev, result.nfev)