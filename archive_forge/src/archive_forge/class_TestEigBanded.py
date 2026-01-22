import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
class TestEigBanded:

    def setup_method(self):
        self.create_bandmat()

    def create_bandmat(self):
        """Create the full matrix `self.fullmat` and
           the corresponding band matrix `self.bandmat`."""
        N = 10
        self.KL = 2
        self.KU = 2
        self.sym_mat = diag(full(N, 1.0)) + diag(full(N - 1, -1.0), -1) + diag(full(N - 1, -1.0), 1) + diag(full(N - 2, -2.0), -2) + diag(full(N - 2, -2.0), 2)
        self.herm_mat = diag(full(N, -1.0)) + 1j * diag(full(N - 1, 1.0), -1) - 1j * diag(full(N - 1, 1.0), 1) + diag(full(N - 2, -2.0), -2) + diag(full(N - 2, -2.0), 2)
        self.real_mat = diag(full(N, 1.0)) + diag(full(N - 1, -1.0), -1) + diag(full(N - 1, -3.0), 1) + diag(full(N - 2, 2.0), -2) + diag(full(N - 2, -2.0), 2)
        self.comp_mat = 1j * diag(full(N, 1.0)) + diag(full(N - 1, -1.0), -1) + 1j * diag(full(N - 1, -3.0), 1) + diag(full(N - 2, 2.0), -2) + diag(full(N - 2, -2.0), 2)
        ew, ev = linalg.eig(self.sym_mat)
        ew = ew.real
        args = argsort(ew)
        self.w_sym_lin = ew[args]
        self.evec_sym_lin = ev[:, args]
        ew, ev = linalg.eig(self.herm_mat)
        ew = ew.real
        args = argsort(ew)
        self.w_herm_lin = ew[args]
        self.evec_herm_lin = ev[:, args]
        LDAB = self.KU + 1
        self.bandmat_sym = zeros((LDAB, N), dtype=float)
        self.bandmat_herm = zeros((LDAB, N), dtype=complex)
        for i in range(LDAB):
            self.bandmat_sym[LDAB - i - 1, i:N] = diag(self.sym_mat, i)
            self.bandmat_herm[LDAB - i - 1, i:N] = diag(self.herm_mat, i)
        LDAB = 2 * self.KL + self.KU + 1
        self.bandmat_real = zeros((LDAB, N), dtype=float)
        self.bandmat_real[2 * self.KL, :] = diag(self.real_mat)
        for i in range(self.KL):
            self.bandmat_real[2 * self.KL - 1 - i, i + 1:N] = diag(self.real_mat, i + 1)
            self.bandmat_real[2 * self.KL + 1 + i, 0:N - 1 - i] = diag(self.real_mat, -i - 1)
        self.bandmat_comp = zeros((LDAB, N), dtype=complex)
        self.bandmat_comp[2 * self.KL, :] = diag(self.comp_mat)
        for i in range(self.KL):
            self.bandmat_comp[2 * self.KL - 1 - i, i + 1:N] = diag(self.comp_mat, i + 1)
            self.bandmat_comp[2 * self.KL + 1 + i, 0:N - 1 - i] = diag(self.comp_mat, -i - 1)
        self.b = 1.0 * arange(N)
        self.bc = self.b * (1 + 1j)

    def test_dsbev(self):
        """Compare dsbev eigenvalues and eigenvectors with
           the result of linalg.eig."""
        w, evec, info = dsbev(self.bandmat_sym, compute_v=1)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_dsbevd(self):
        """Compare dsbevd eigenvalues and eigenvectors with
           the result of linalg.eig."""
        w, evec, info = dsbevd(self.bandmat_sym, compute_v=1)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_dsbevx(self):
        """Compare dsbevx eigenvalues and eigenvectors
           with the result of linalg.eig."""
        N, N = shape(self.sym_mat)
        w, evec, num, ifail, info = dsbevx(self.bandmat_sym, 0.0, 0.0, 1, N, compute_v=1, range=2)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_zhbevd(self):
        """Compare zhbevd eigenvalues and eigenvectors
           with the result of linalg.eig."""
        w, evec, info = zhbevd(self.bandmat_herm, compute_v=1)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_herm_lin))

    def test_zhbevx(self):
        """Compare zhbevx eigenvalues and eigenvectors
           with the result of linalg.eig."""
        N, N = shape(self.herm_mat)
        w, evec, num, ifail, info = zhbevx(self.bandmat_herm, 0.0, 0.0, 1, N, compute_v=1, range=2)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_herm_lin))

    def test_eigvals_banded(self):
        """Compare eigenvalues of eigvals_banded with those of linalg.eig."""
        w_sym = eigvals_banded(self.bandmat_sym)
        w_sym = w_sym.real
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)
        w_herm = eigvals_banded(self.bandmat_herm)
        w_herm = w_herm.real
        assert_array_almost_equal(sort(w_herm), self.w_herm_lin)
        ind1 = 2
        ind2 = np.longlong(6)
        w_sym_ind = eigvals_banded(self.bandmat_sym, select='i', select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_sym_ind), self.w_sym_lin[ind1:ind2 + 1])
        w_herm_ind = eigvals_banded(self.bandmat_herm, select='i', select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_herm_ind), self.w_herm_lin[ind1:ind2 + 1])
        v_lower = self.w_sym_lin[ind1] - 1e-05
        v_upper = self.w_sym_lin[ind2] + 1e-05
        w_sym_val = eigvals_banded(self.bandmat_sym, select='v', select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_sym_val), self.w_sym_lin[ind1:ind2 + 1])
        v_lower = self.w_herm_lin[ind1] - 1e-05
        v_upper = self.w_herm_lin[ind2] + 1e-05
        w_herm_val = eigvals_banded(self.bandmat_herm, select='v', select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_herm_val), self.w_herm_lin[ind1:ind2 + 1])
        w_sym = eigvals_banded(self.bandmat_sym, check_finite=False)
        w_sym = w_sym.real
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)

    def test_eig_banded(self):
        """Compare eigenvalues and eigenvectors of eig_banded
           with those of linalg.eig. """
        w_sym, evec_sym = eig_banded(self.bandmat_sym)
        evec_sym_ = evec_sym[:, argsort(w_sym.real)]
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_sym_), abs(self.evec_sym_lin))
        w_herm, evec_herm = eig_banded(self.bandmat_herm)
        evec_herm_ = evec_herm[:, argsort(w_herm.real)]
        assert_array_almost_equal(sort(w_herm), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_herm_), abs(self.evec_herm_lin))
        ind1 = 2
        ind2 = 6
        w_sym_ind, evec_sym_ind = eig_banded(self.bandmat_sym, select='i', select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_sym_ind), self.w_sym_lin[ind1:ind2 + 1])
        assert_array_almost_equal(abs(evec_sym_ind), abs(self.evec_sym_lin[:, ind1:ind2 + 1]))
        w_herm_ind, evec_herm_ind = eig_banded(self.bandmat_herm, select='i', select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_herm_ind), self.w_herm_lin[ind1:ind2 + 1])
        assert_array_almost_equal(abs(evec_herm_ind), abs(self.evec_herm_lin[:, ind1:ind2 + 1]))
        v_lower = self.w_sym_lin[ind1] - 1e-05
        v_upper = self.w_sym_lin[ind2] + 1e-05
        w_sym_val, evec_sym_val = eig_banded(self.bandmat_sym, select='v', select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_sym_val), self.w_sym_lin[ind1:ind2 + 1])
        assert_array_almost_equal(abs(evec_sym_val), abs(self.evec_sym_lin[:, ind1:ind2 + 1]))
        v_lower = self.w_herm_lin[ind1] - 1e-05
        v_upper = self.w_herm_lin[ind2] + 1e-05
        w_herm_val, evec_herm_val = eig_banded(self.bandmat_herm, select='v', select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_herm_val), self.w_herm_lin[ind1:ind2 + 1])
        assert_array_almost_equal(abs(evec_herm_val), abs(self.evec_herm_lin[:, ind1:ind2 + 1]))
        w_sym, evec_sym = eig_banded(self.bandmat_sym, check_finite=False)
        evec_sym_ = evec_sym[:, argsort(w_sym.real)]
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_sym_), abs(self.evec_sym_lin))

    def test_dgbtrf(self):
        """Compare dgbtrf  LU factorisation with the LU factorisation result
           of linalg.lu."""
        M, N = shape(self.real_mat)
        lu_symm_band, ipiv, info = dgbtrf(self.bandmat_real, self.KL, self.KU)
        u = diag(lu_symm_band[2 * self.KL, :])
        for i in range(self.KL + self.KU):
            u += diag(lu_symm_band[2 * self.KL - 1 - i, i + 1:N], i + 1)
        p_lin, l_lin, u_lin = lu(self.real_mat, permute_l=0)
        assert_array_almost_equal(u, u_lin)

    def test_zgbtrf(self):
        """Compare zgbtrf  LU factorisation with the LU factorisation result
           of linalg.lu."""
        M, N = shape(self.comp_mat)
        lu_symm_band, ipiv, info = zgbtrf(self.bandmat_comp, self.KL, self.KU)
        u = diag(lu_symm_band[2 * self.KL, :])
        for i in range(self.KL + self.KU):
            u += diag(lu_symm_band[2 * self.KL - 1 - i, i + 1:N], i + 1)
        p_lin, l_lin, u_lin = lu(self.comp_mat, permute_l=0)
        assert_array_almost_equal(u, u_lin)

    def test_dgbtrs(self):
        """Compare dgbtrs  solutions for linear equation system  A*x = b
           with solutions of linalg.solve."""
        lu_symm_band, ipiv, info = dgbtrf(self.bandmat_real, self.KL, self.KU)
        y, info = dgbtrs(lu_symm_band, self.KL, self.KU, self.b, ipiv)
        y_lin = linalg.solve(self.real_mat, self.b)
        assert_array_almost_equal(y, y_lin)

    def test_zgbtrs(self):
        """Compare zgbtrs  solutions for linear equation system  A*x = b
           with solutions of linalg.solve."""
        lu_symm_band, ipiv, info = zgbtrf(self.bandmat_comp, self.KL, self.KU)
        y, info = zgbtrs(lu_symm_band, self.KL, self.KU, self.bc, ipiv)
        y_lin = linalg.solve(self.comp_mat, self.bc)
        assert_array_almost_equal(y, y_lin)