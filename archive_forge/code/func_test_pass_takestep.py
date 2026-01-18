import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_pass_takestep(self):
    takestep = MyTakeStep1()
    initial_step_size = takestep.stepsize
    i = 1
    res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=self.niter, disp=self.disp, take_step=takestep)
    assert_almost_equal(res.x, self.sol[i], self.tol)
    assert_(takestep.been_called)
    assert_(initial_step_size != takestep.stepsize)