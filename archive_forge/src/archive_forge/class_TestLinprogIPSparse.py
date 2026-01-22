import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
class TestLinprogIPSparse(LinprogIPTests):
    options = {'sparse': True, 'cholesky': False, 'sym_pos': False}

    @pytest.mark.xfail_on_32bit('This test is sensitive to machine epsilon level perturbations in linear system solution in _linprog_ip._sym_solve.')
    def test_bug_6139(self):
        super().test_bug_6139()

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        super().test_bug_6690()

    def test_magic_square_sparse_no_presolve(self):
        A_eq, b_eq, c, _, _ = magic_square(3)
        bounds = (0, 1)
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(MatrixRankWarning, 'Matrix is exactly singular')
            sup.filter(OptimizeWarning, 'Solving system with option...')
            o = {key: self.options[key] for key in self.options}
            o['presolve'] = False
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
        _assert_success(res, desired_fun=1.730550597)

    def test_sparse_solve_options(self):
        A_eq, b_eq, c, _, _ = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            sup.filter(OptimizeWarning, 'Invalid permc_spec option')
            o = {key: self.options[key] for key in self.options}
            permc_specs = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD', 'ekki-ekki-ekki')
            for permc_spec in permc_specs:
                o['permc_spec'] = permc_spec
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
                _assert_success(res, desired_fun=1.730550597)