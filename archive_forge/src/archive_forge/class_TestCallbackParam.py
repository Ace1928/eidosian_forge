import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
class TestCallbackParam(BaseTest):
    x = cp.Variable()
    p = cp.Parameter()
    q = cp.Parameter()

    def test_callback_param(self) -> None:
        callback_param = cp.CallbackParam(callback=lambda: self.p.value * self.q.value)
        problem = cp.Problem(cp.Minimize(self.x), [self.x >= callback_param])
        assert problem.is_dpp()
        self.p.value = 1.0
        self.q.value = 4.0
        problem.solve()
        self.assertAlmostEqual(self.x.value, 4.0)
        self.p.value = 2.0
        problem.solve()
        self.assertAlmostEqual(self.x.value, 8.0)
        with pytest.raises(NotImplementedError, match='Cannot set the value of a CallbackParam'):
            callback_param.value = 1.0