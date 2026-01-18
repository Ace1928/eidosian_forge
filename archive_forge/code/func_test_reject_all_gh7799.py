import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_reject_all_gh7799(self):

    def fun(x):
        return x @ x

    def constraint(x):
        return x + 1
    kwargs = {'constraints': {'type': 'eq', 'fun': constraint}, 'bounds': [(0, 1), (0, 1)], 'method': 'slsqp'}
    res = basinhopping(fun, x0=[2, 3], niter=10, minimizer_kwargs=kwargs)
    assert not res.success