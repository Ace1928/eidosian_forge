import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
class TestAdjustSchemeToBounds:

    def test_no_bounds(self):
        x0 = np.zeros(3)
        h = np.full(3, 0.01)
        inf_lower = np.empty_like(x0)
        inf_upper = np.empty_like(x0)
        inf_lower.fill(-np.inf)
        inf_upper.fill(np.inf)
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(np.all(one_sided))
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(np.all(one_sided))
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 1, '2-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(np.all(~one_sided))
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 2, '2-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(np.all(~one_sided))

    def test_with_bound(self):
        x0 = np.array([0.0, 0.85, -0.85])
        lb = -np.ones(3)
        ub = np.ones(3)
        h = np.array([1, 1, -1]) * 0.1
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
        assert_allclose(h_adjusted, h)
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
        assert_allclose(h_adjusted, np.array([1, -1, 1]) * 0.1)
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 1, '2-sided', lb, ub)
        assert_allclose(h_adjusted, np.abs(h))
        assert_(np.all(~one_sided))
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 2, '2-sided', lb, ub)
        assert_allclose(h_adjusted, np.array([1, -1, 1]) * 0.1)
        assert_equal(one_sided, np.array([False, True, True]))

    def test_tight_bounds(self):
        lb = np.array([-0.03, -0.03])
        ub = np.array([0.05, 0.05])
        x0 = np.array([0.0, 0.03])
        h = np.array([-0.1, -0.1])
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
        assert_allclose(h_adjusted, np.array([0.05, -0.06]))
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
        assert_allclose(h_adjusted, np.array([0.025, -0.03]))
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 1, '2-sided', lb, ub)
        assert_allclose(h_adjusted, np.array([0.03, -0.03]))
        assert_equal(one_sided, np.array([False, True]))
        h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 2, '2-sided', lb, ub)
        assert_allclose(h_adjusted, np.array([0.015, -0.015]))
        assert_equal(one_sided, np.array([False, True]))