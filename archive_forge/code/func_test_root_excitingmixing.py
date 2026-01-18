from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_root_excitingmixing(self):
    res = root(F, F.xin, method='excitingmixing', options={'nit': 20, 'jac_options': {'alpha': 0.5}})
    assert_(nonlin.norm(res.x) < 1e-05)
    assert_(nonlin.norm(res.fun) < 1e-05)