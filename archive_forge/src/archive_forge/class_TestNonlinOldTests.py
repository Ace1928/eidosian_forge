from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
class TestNonlinOldTests:
    """ Test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    def test_broyden1(self):
        x = nonlin.broyden1(F, F.xin, iter=12, alpha=1)
        assert_(nonlin.norm(x) < 1e-09)
        assert_(nonlin.norm(F(x)) < 1e-09)

    def test_broyden2(self):
        x = nonlin.broyden2(F, F.xin, iter=12, alpha=1)
        assert_(nonlin.norm(x) < 1e-09)
        assert_(nonlin.norm(F(x)) < 1e-09)

    def test_anderson(self):
        x = nonlin.anderson(F, F.xin, iter=12, alpha=0.03, M=5)
        assert_(nonlin.norm(x) < 0.33)

    def test_linearmixing(self):
        x = nonlin.linearmixing(F, F.xin, iter=60, alpha=0.5)
        assert_(nonlin.norm(x) < 1e-07)
        assert_(nonlin.norm(F(x)) < 1e-07)

    def test_exciting(self):
        x = nonlin.excitingmixing(F, F.xin, iter=20, alpha=0.5)
        assert_(nonlin.norm(x) < 1e-05)
        assert_(nonlin.norm(F(x)) < 1e-05)

    def test_diagbroyden(self):
        x = nonlin.diagbroyden(F, F.xin, iter=11, alpha=1)
        assert_(nonlin.norm(x) < 1e-08)
        assert_(nonlin.norm(F(x)) < 1e-08)

    def test_root_broyden1(self):
        res = root(F, F.xin, method='broyden1', options={'nit': 12, 'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-09)
        assert_(nonlin.norm(res.fun) < 1e-09)

    def test_root_broyden2(self):
        res = root(F, F.xin, method='broyden2', options={'nit': 12, 'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-09)
        assert_(nonlin.norm(res.fun) < 1e-09)

    def test_root_anderson(self):
        res = root(F, F.xin, method='anderson', options={'nit': 12, 'jac_options': {'alpha': 0.03, 'M': 5}})
        assert_(nonlin.norm(res.x) < 0.33)

    def test_root_linearmixing(self):
        res = root(F, F.xin, method='linearmixing', options={'nit': 60, 'jac_options': {'alpha': 0.5}})
        assert_(nonlin.norm(res.x) < 1e-07)
        assert_(nonlin.norm(res.fun) < 1e-07)

    def test_root_excitingmixing(self):
        res = root(F, F.xin, method='excitingmixing', options={'nit': 20, 'jac_options': {'alpha': 0.5}})
        assert_(nonlin.norm(res.x) < 1e-05)
        assert_(nonlin.norm(res.fun) < 1e-05)

    def test_root_diagbroyden(self):
        res = root(F, F.xin, method='diagbroyden', options={'nit': 11, 'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-08)
        assert_(nonlin.norm(res.fun) < 1e-08)