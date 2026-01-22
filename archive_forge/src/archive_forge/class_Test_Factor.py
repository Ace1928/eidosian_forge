import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
class Test_Factor:

    def test_corr_nearest_factor_arrpack(self):
        u2 = np.array([[6.39407581e-19, 0.00915225947, 0.0182631698, 0.0272917181, 0.0361975557, 0.0449413101, 0.0534848732, 0.0617916613, 0.0698268388, 0.0775575058, 0.0849528448, 0.0919842264, 0.0986252769, 0.104851906, 0.110642305, 0.115976906, 0.120838331, 0.125211306, 0.12908257, 0.132440778, 0.135276397, 0.137581605, 0.139350201, 0.140577526, 0.141260396, 0.141397057, 0.14098716, 0.140031756, 0.138533306, 0.136495727, 0.133924439, 0.130826443, 0.127210404, 0.12308675, 0.118467769, 0.113367717, 0.107802909, 0.101791811, 0.0953551023, 0.088515732, 0.0812989329, 0.0737322125, 0.0658453049, 0.0576700847, 0.0492404406, 0.0405921079, 0.0317624629, 0.0227902803, 0.0137154584, 0.00457871801, -0.00457871801, -0.0137154584, -0.0227902803, -0.0317624629, -0.0405921079, -0.0492404406, -0.0576700847, -0.0658453049, -0.0737322125, -0.0812989329, -0.088515732, -0.0953551023, -0.101791811, -0.107802909, -0.113367717, -0.118467769, -0.12308675, -0.127210404, -0.130826443, -0.133924439, -0.136495727, -0.138533306, -0.140031756, -0.14098716, -0.141397057, -0.141260396, -0.140577526, -0.139350201, -0.137581605, -0.135276397, -0.132440778, -0.12908257, -0.125211306, -0.120838331, -0.115976906, -0.110642305, -0.104851906, -0.0986252769, -0.0919842264, -0.0849528448, -0.0775575058, -0.0698268388, -0.0617916613, -0.0534848732, -0.0449413101, -0.0361975557, -0.0272917181, -0.0182631698, -0.00915225947, -3.51829569e-17]]).T
        s2 = np.array([24.88812183])
        d = 100
        dm = 1
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2 * np.pi, d)
        for j in range(dm):
            X[:, j] = np.sin(x * (j + 1))
        _project_correlation_factors(X)
        X *= 0.7
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, 1.0)
        from scipy.sparse.linalg import svds
        u, s, vt = svds(mat, dm)
        dsign = np.sign(u[1]) * np.sign(u2[1])
        assert_allclose(u, dsign * u2, rtol=1e-06, atol=1e-14)
        assert_allclose(s, s2, rtol=1e-06)

    @pytest.mark.parametrize('dm', [1, 2])
    def test_corr_nearest_factor(self, dm):
        objvals = [np.array([6241.8, 6241.8, 579.4, 264.6, 264.3]), np.array([2104.9, 2104.9, 710.5, 266.3, 286.1])]
        d = 100
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2 * np.pi, d)
        np.random.seed(10)
        for j in range(dm):
            X[:, j] = np.sin(x * (j + 1)) + 1e-10 * np.random.randn(d)
        _project_correlation_factors(X)
        assert np.isfinite(X).all()
        X *= 0.7
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, 1.0)
        rslt = corr_nearest_factor(mat, dm, maxiter=10000)
        err_msg = 'rank=%d, niter=%d' % (dm, len(rslt.objective_values))
        assert_allclose(rslt.objective_values[:5], objvals[dm - 1], rtol=0.5, err_msg=err_msg)
        assert rslt.Converged
        mat1 = rslt.corr.to_matrix()
        assert_allclose(mat, mat1, rtol=0.25, atol=0.001, err_msg=err_msg)

    @pytest.mark.slow
    @pytest.mark.parametrize('dm', [1, 2])
    def test_corr_nearest_factor_sparse(self, dm):
        d = 200
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2 * np.pi, d)
        rs = np.random.RandomState(10)
        for j in range(dm):
            X[:, j] = np.sin(x * (j + 1)) + rs.randn(d)
        _project_correlation_factors(X)
        X *= 0.7
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, 1)
        mat.flat[np.abs(mat.flat) < 0.35] = 0.0
        smat = sparse.csr_matrix(mat)
        dense_rslt = corr_nearest_factor(mat, dm, maxiter=10000)
        sparse_rslt = corr_nearest_factor(smat, dm, maxiter=10000)
        mat_dense = dense_rslt.corr.to_matrix()
        mat_sparse = sparse_rslt.corr.to_matrix()
        assert dense_rslt.Converged is sparse_rslt.Converged
        assert dense_rslt.Converged is True
        assert_allclose(mat_dense, mat_sparse, rtol=0.25, atol=0.001)

    def test_spg_optim(self, reset_randomstate):
        dm = 100
        ind = np.arange(dm)
        indmat = np.abs(ind[:, None] - ind[None, :])
        M = 0.8 ** indmat

        def obj(x):
            return np.dot(x, np.dot(M, x))

        def grad(x):
            return 2 * np.dot(M, x)

        def project(x):
            return x
        x = np.random.normal(size=dm)
        rslt = _spg_optim(obj, grad, x, project)
        xnew = rslt.params
        assert rslt.Converged is True
        assert_almost_equal(obj(xnew), 0, decimal=3)

    def test_decorrelate(self, reset_randomstate):
        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 4))
        fac = FactoredPSDMatrix(dg, root)
        mat = fac.to_matrix()
        rmat = np.linalg.cholesky(mat)
        dcr = fac.decorrelate(rmat)
        idm = np.dot(dcr, dcr.T)
        assert_almost_equal(idm, np.eye(d))
        rhs = np.random.normal(size=(d, 5))
        mat2 = np.dot(rhs.T, np.linalg.solve(mat, rhs))
        mat3 = fac.decorrelate(rhs)
        mat3 = np.dot(mat3.T, mat3)
        assert_almost_equal(mat2, mat3)

    def test_logdet(self, reset_randomstate):
        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 4))
        fac = FactoredPSDMatrix(dg, root)
        mat = fac.to_matrix()
        _, ld = np.linalg.slogdet(mat)
        ld2 = fac.logdet()
        assert_almost_equal(ld, ld2)

    def test_solve(self, reset_randomstate):
        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 2))
        fac = FactoredPSDMatrix(dg, root)
        rhs = np.random.normal(size=(d, 5))
        sr1 = fac.solve(rhs)
        mat = fac.to_matrix()
        sr2 = np.linalg.solve(mat, rhs)
        assert_almost_equal(sr1, sr2)

    @pytest.mark.parametrize('dm', [1, 2])
    def test_cov_nearest_factor_homog(self, dm):
        d = 100
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2 * np.pi, d)
        for j in range(dm):
            X[:, j] = np.sin(x * (j + 1))
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, np.diag(mat) + 3.1)
        rslt = cov_nearest_factor_homog(mat, dm)
        mat1 = rslt.to_matrix()
        assert_allclose(mat, mat1, rtol=0.25, atol=0.001)

    @pytest.mark.parametrize('dm', [1, 2])
    def test_cov_nearest_factor_homog_sparse(self, dm):
        d = 100
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2 * np.pi, d)
        for j in range(dm):
            X[:, j] = np.sin(x * (j + 1))
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, np.diag(mat) + 3.1)
        rslt = cov_nearest_factor_homog(mat, dm)
        mat1 = rslt.to_matrix()
        smat = sparse.csr_matrix(mat)
        rslt = cov_nearest_factor_homog(smat, dm)
        mat2 = rslt.to_matrix()
        assert_allclose(mat1, mat2, rtol=0.25, atol=0.001)

    def test_corr_thresholded(self, reset_randomstate):
        import datetime
        t1 = datetime.datetime.now()
        X = np.random.normal(size=(2000, 10))
        tcor = corr_thresholded(X, 0.2, max_elt=4000000.0)
        t2 = datetime.datetime.now()
        ss = (t2 - t1).seconds
        fcor = np.corrcoef(X)
        fcor *= np.abs(fcor) >= 0.2
        assert_allclose(tcor.todense(), fcor, rtol=0.25, atol=0.001)