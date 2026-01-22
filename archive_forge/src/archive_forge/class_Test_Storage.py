import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
class Test_Storage:

    def setup_method(self):
        self.x0 = np.array(1)
        self.f0 = 0
        minres = OptimizeResult(success=True)
        minres.x = self.x0
        minres.fun = self.f0
        self.storage = Storage(minres)

    def test_higher_f_rejected(self):
        new_minres = OptimizeResult(success=True)
        new_minres.x = self.x0 + 1
        new_minres.fun = self.f0 + 1
        ret = self.storage.update(new_minres)
        minres = self.storage.get_lowest()
        assert_equal(self.x0, minres.x)
        assert_equal(self.f0, minres.fun)
        assert_(not ret)

    @pytest.mark.parametrize('success', [True, False])
    def test_lower_f_accepted(self, success):
        new_minres = OptimizeResult(success=success)
        new_minres.x = self.x0 + 1
        new_minres.fun = self.f0 - 1
        ret = self.storage.update(new_minres)
        minres = self.storage.get_lowest()
        assert (self.x0 != minres.x) == success
        assert (self.f0 != minres.fun) == success
        assert ret is success