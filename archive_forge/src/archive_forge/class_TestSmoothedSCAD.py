import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess
class TestSmoothedSCAD(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.SCADSmoothed(tau=0.05, c0=0.05)