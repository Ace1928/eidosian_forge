from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def _check_dot(self, jac_cls, complex=False, tol=1e-06, **kw):
    np.random.seed(123)
    N = 7

    def rand(*a):
        q = np.random.rand(*a)
        if complex:
            q = q + 1j * np.random.rand(*a)
        return q

    def assert_close(a, b, msg):
        d = abs(a - b).max()
        f = tol + abs(b).max() * tol
        if d > f:
            raise AssertionError(f'{msg}: err {d:g}')
    self.A = rand(N, N)
    x0 = np.random.rand(N)
    jac = jac_cls(**kw)
    jac.setup(x0, self._func(x0), self._func)
    for k in range(2 * N):
        v = rand(N)
        if hasattr(jac, '__array__'):
            Jd = np.array(jac)
            if hasattr(jac, 'solve'):
                Gv = jac.solve(v)
                Gv2 = np.linalg.solve(Jd, v)
                assert_close(Gv, Gv2, 'solve vs array')
            if hasattr(jac, 'rsolve'):
                Gv = jac.rsolve(v)
                Gv2 = np.linalg.solve(Jd.T.conj(), v)
                assert_close(Gv, Gv2, 'rsolve vs array')
            if hasattr(jac, 'matvec'):
                Jv = jac.matvec(v)
                Jv2 = np.dot(Jd, v)
                assert_close(Jv, Jv2, 'dot vs array')
            if hasattr(jac, 'rmatvec'):
                Jv = jac.rmatvec(v)
                Jv2 = np.dot(Jd.T.conj(), v)
                assert_close(Jv, Jv2, 'rmatvec vs array')
        if hasattr(jac, 'matvec') and hasattr(jac, 'solve'):
            Jv = jac.matvec(v)
            Jv2 = jac.solve(jac.matvec(Jv))
            assert_close(Jv, Jv2, 'dot vs solve')
        if hasattr(jac, 'rmatvec') and hasattr(jac, 'rsolve'):
            Jv = jac.rmatvec(v)
            Jv2 = jac.rmatvec(jac.rsolve(Jv))
            assert_close(Jv, Jv2, 'rmatvec vs rsolve')
        x = rand(N)
        jac.update(x, self._func(x))