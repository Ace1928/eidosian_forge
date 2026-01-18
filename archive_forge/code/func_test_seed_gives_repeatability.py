import multiprocessing
import platform
from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
from scipy.optimize import rosen, minimize
from scipy.sparse import csr_matrix
from scipy import stats
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises, warns
import pytest
def test_seed_gives_repeatability(self):
    result = differential_evolution(self.quadratic, [(-100, 100)], polish=False, seed=1, tol=0.5)
    result2 = differential_evolution(self.quadratic, [(-100, 100)], polish=False, seed=1, tol=0.5)
    assert_equal(result.x, result2.x)
    assert_equal(result.nfev, result2.nfev)