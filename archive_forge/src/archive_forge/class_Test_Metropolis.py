import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
class Test_Metropolis:

    def setup_method(self):
        self.T = 2.0
        self.met = Metropolis(self.T)
        self.res_new = OptimizeResult(success=True, fun=0.0)
        self.res_old = OptimizeResult(success=True, fun=1.0)

    def test_boolean_return(self):
        ret = self.met(res_new=self.res_new, res_old=self.res_old)
        assert isinstance(ret, bool)

    def test_lower_f_accepted(self):
        assert_(self.met(res_new=self.res_new, res_old=self.res_old))

    def test_accept(self):
        one_accept = False
        one_reject = False
        for i in range(1000):
            if one_accept and one_reject:
                break
            res_new = OptimizeResult(success=True, fun=1.0)
            res_old = OptimizeResult(success=True, fun=0.5)
            ret = self.met(res_new=res_new, res_old=res_old)
            if ret:
                one_accept = True
            else:
                one_reject = True
        assert_(one_accept)
        assert_(one_reject)

    def test_GH7495(self):
        met = Metropolis(2)
        res_new = OptimizeResult(success=True, fun=0.0)
        res_old = OptimizeResult(success=True, fun=2000)
        with np.errstate(over='raise'):
            met.accept_reject(res_new=res_new, res_old=res_old)

    def test_gh7799(self):

        def func(x):
            return (x ** 2 - 8) ** 2 + (x + 2) ** 2
        x0 = -4
        limit = 50
        con = ({'type': 'ineq', 'fun': lambda x: func(x) - limit},)
        res = basinhopping(func, x0, 30, minimizer_kwargs={'constraints': con})
        assert res.success
        assert_allclose(res.fun, limit, rtol=1e-06)

    def test_accept_gh7799(self):
        met = Metropolis(0)
        res_new = OptimizeResult(success=True, fun=0.0)
        res_old = OptimizeResult(success=True, fun=1.0)
        assert met(res_new=res_new, res_old=res_old)
        res_new.success = False
        assert not met(res_new=res_new, res_old=res_old)
        res_old.success = False
        assert met(res_new=res_new, res_old=res_old)

    def test_reject_all_gh7799(self):

        def fun(x):
            return x @ x

        def constraint(x):
            return x + 1
        kwargs = {'constraints': {'type': 'eq', 'fun': constraint}, 'bounds': [(0, 1), (0, 1)], 'method': 'slsqp'}
        res = basinhopping(fun, x0=[2, 3], niter=10, minimizer_kwargs=kwargs)
        assert not res.success