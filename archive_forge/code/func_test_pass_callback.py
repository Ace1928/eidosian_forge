import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_pass_callback(self):
    callback = MyCallBack()
    i = 1
    res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=30, disp=self.disp, callback=callback)
    assert_(callback.been_called)
    assert_('callback' in res.message[0])
    assert_equal(res.nit, 9)