import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
class TestDerivativeFun(CheckDerivativeMixin):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.args = (cls.x,)

    def fun(self):
        return fun

    def gradtrue(self, params):
        return self.x.sum(0)

    def hesstrue(self, params):
        return np.zeros((3, 3))