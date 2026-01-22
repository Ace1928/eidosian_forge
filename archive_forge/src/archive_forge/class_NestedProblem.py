from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
class NestedProblem:

    def __init__(self):
        self.F_outer_count = 0

    def F_outer(self, x):
        self.F_outer_count += 1
        if self.F_outer_count > 1000:
            raise Exception('Nested minimization failed to terminate.')
        inner_res = minimize(self.F_inner, (3, 4), method='SLSQP')
        assert_(inner_res.success)
        assert_allclose(inner_res.x, [1, 1])
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def F_inner(self, x):
        return (x[0] - 1) ** 2 + (x[1] - 1) ** 2

    def solve(self):
        outer_res = minimize(self.F_outer, (5, 5, 5), method='SLSQP')
        assert_(outer_res.success)
        assert_allclose(outer_res.x, [0, 0, 0])