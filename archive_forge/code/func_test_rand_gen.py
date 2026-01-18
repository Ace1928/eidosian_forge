from scipy.optimize import dual_annealing, Bounds
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
from scipy._lib._util import check_random_state
def test_rand_gen(self):
    rng = np.random.default_rng(1)
    res1 = dual_annealing(self.func, self.ld_bounds, seed=rng)
    rng = np.random.default_rng(1)
    res2 = dual_annealing(self.func, self.ld_bounds, seed=rng)
    assert_equal(res1.x, res2.x)