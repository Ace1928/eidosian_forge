import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_pass_accept_test(self):
    accept_test = MyAcceptTest()
    i = 1
    basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs, niter=10, disp=self.disp, accept_test=accept_test)
    assert_(accept_test.been_called)