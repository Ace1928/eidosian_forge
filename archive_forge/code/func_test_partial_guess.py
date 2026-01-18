import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
def test_partial_guess(self):
    n = 5
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    res1 = quadratic_assignment(A, B, method=self.method, options={'rng': 0})
    guess = np.array([np.arange(5), res1.col_ind]).T
    res2 = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'partial_guess': guess})
    fix = [2, 4]
    match = np.array([np.arange(5)[fix], res1.col_ind[fix]]).T
    res3 = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'partial_guess': guess, 'partial_match': match})
    assert_(res1.nit != n * (n + 1) / 2)
    assert_equal(res2.nit, n * (n + 1) / 2)
    assert_equal(res3.nit, (n - 2) * (n - 1) / 2)