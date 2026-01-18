from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_root_broyden2(self):
    res = root(F, F.xin, method='broyden2', options={'nit': 12, 'jac_options': {'alpha': 1}})
    assert_(nonlin.norm(res.x) < 1e-09)
    assert_(nonlin.norm(res.fun) < 1e-09)