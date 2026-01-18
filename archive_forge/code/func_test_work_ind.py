import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_work_ind(self):

    def func(par, x):
        b0, b1 = par
        return b0 + b1 * x
    n_data = 4
    x = np.arange(n_data)
    y = np.where(x % 2, x + 0.1, x - 0.1)
    x_err = np.full(n_data, 0.1)
    y_err = np.full(n_data, 0.1)
    linear_model = Model(func)
    real_data = RealData(x, y, sx=x_err, sy=y_err)
    odr_obj = ODR(real_data, linear_model, beta0=[0.4, 0.4])
    odr_obj.set_job(fit_type=0)
    out = odr_obj.run()
    sd_ind = out.work_ind['sd']
    assert_array_almost_equal(out.sd_beta, out.work[sd_ind:sd_ind + len(out.sd_beta)])