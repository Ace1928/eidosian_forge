import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_seed_reproducibility(self):
    minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True}
    f_1 = []

    def callback(x, f, accepted):
        f_1.append(f)
    basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, callback=callback, seed=10)
    f_2 = []

    def callback2(x, f, accepted):
        f_2.append(f)
    basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, callback=callback2, seed=10)
    assert_equal(np.array(f_1), np.array(f_2))