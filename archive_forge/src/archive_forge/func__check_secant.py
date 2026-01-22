from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def _check_secant(self, jac_cls, npoints=1, **kw):
    """
        Check that the given Jacobian approximation satisfies secant
        conditions for last `npoints` points.
        """
    jac = jac_cls(**kw)
    jac.setup(self.xs[0], self.fs[0], None)
    for j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
        jac.update(x, f)
        for k in range(min(npoints, j + 1)):
            dx = self.xs[j - k + 1] - self.xs[j - k]
            df = self.fs[j - k + 1] - self.fs[j - k]
            assert_(np.allclose(dx, jac.solve(df)))
        if j >= npoints:
            dx = self.xs[j - npoints + 1] - self.xs[j - npoints]
            df = self.fs[j - npoints + 1] - self.fs[j - npoints]
            assert_(not np.allclose(dx, jac.solve(df)))