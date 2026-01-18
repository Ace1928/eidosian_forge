from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize('method', ['lgmres', 'gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr'])
def test_tol_norm_called(self, method):
    self._tol_norm_used = False

    def local_norm_func(x):
        self._tol_norm_used = True
        return np.absolute(x).max()
    nonlin.newton_krylov(F, F.xin, method=method, f_tol=0.01, maxiter=200, verbose=0, tol_norm=local_norm_func)
    assert_(self._tol_norm_used)