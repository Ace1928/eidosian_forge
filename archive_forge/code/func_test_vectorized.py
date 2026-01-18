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
def test_vectorized(self):

    def quadratic(x):
        return np.sum(x ** 2)

    def quadratic_vec(x):
        return np.sum(x ** 2, axis=0)
    with pytest.raises(RuntimeError, match='The vectorized function'):
        differential_evolution(quadratic, self.bounds, vectorized=True, updating='deferred')
    with warns(UserWarning, match="differential_evolution: the 'vector"):
        differential_evolution(quadratic_vec, self.bounds, vectorized=True)
    with warns(UserWarning, match="differential_evolution: the 'workers"):
        differential_evolution(quadratic_vec, self.bounds, vectorized=True, workers=map, updating='deferred')
    ncalls = [0]

    def rosen_vec(x):
        ncalls[0] += 1
        return rosen(x)
    bounds = [(0, 10), (0, 10)]
    res1 = differential_evolution(rosen, bounds, updating='deferred', seed=1)
    res2 = differential_evolution(rosen_vec, bounds, vectorized=True, updating='deferred', seed=1)
    assert_allclose(res1.x, res2.x)
    assert ncalls[0] == res2.nfev
    assert res1.nit == res2.nit