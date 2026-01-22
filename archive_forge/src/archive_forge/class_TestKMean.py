import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
class TestKMean:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_large_features(self, xp):
        d = 300
        n = 100
        m1 = np.random.randn(d)
        m2 = np.random.randn(d)
        x = 10000 * np.random.randn(n, d) - 20000 * m1
        y = 10000 * np.random.randn(n, d) + 20000 * m2
        data = np.empty((x.shape[0] + y.shape[0], d), np.float64)
        data[:x.shape[0]] = x
        data[x.shape[0]:] = y
        kmeans(xp.asarray(data), xp.asarray(2))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans_simple(self, xp):
        np.random.seed(54321)
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
        for tp in arrays:
            code1 = kmeans(tp(X), tp(initc), iter=1)[0]
            xp_assert_close(code1, xp.asarray(CODET2))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans_lost_cluster(self, xp):
        data = xp.asarray(TESTDATA_2D)
        initk = xp.asarray([[-1.8127404, -0.67128041], [2.04621601, 0.07401111], [-2.31149087, -0.05160469]])
        kmeans(data, initk)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'One of the clusters is empty. Re-run kmeans with a different initialization')
            kmeans2(data, initk, missing='warn')
        assert_raises(ClusterError, kmeans2, data, initk, missing='raise')

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_simple(self, xp):
        np.random.seed(12345678)
        initc = xp.asarray(np.concatenate([[X[0]], [X[1]], [X[2]]]))
        arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
        for tp in arrays:
            code1 = kmeans2(tp(X), tp(initc), iter=1)[0]
            code2 = kmeans2(tp(X), tp(initc), iter=2)[0]
            xp_assert_close(code1, xp.asarray(CODET1))
            xp_assert_close(code2, xp.asarray(CODET2))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_rank1(self, xp):
        data = xp.asarray(TESTDATA_2D)
        data1 = data[:, 0]
        initc = data1[:3]
        code = copy(initc, xp=xp)
        kmeans2(data1, code, iter=1)[0]
        kmeans2(data1, code, iter=2)[0]

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_rank1_2(self, xp):
        data = xp.asarray(TESTDATA_2D)
        data1 = data[:, 0]
        kmeans2(data1, xp.asarray(2), iter=1)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_high_dim(self, xp):
        data = xp.asarray(TESTDATA_2D)
        data = xp.reshape(data, (20, 20))[:10, :]
        kmeans2(data, xp.asarray(2))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_init(self, xp):
        np.random.seed(12345)
        data = xp.asarray(TESTDATA_2D)
        k = xp.asarray(3)
        kmeans2(data, k, minit='points')
        kmeans2(data[:, :1], k, minit='points')
        kmeans2(data, k, minit='++')
        kmeans2(data[:, :1], k, minit='++')
        with suppress_warnings() as sup:
            sup.filter(message='One of the clusters is empty. Re-run.')
            kmeans2(data, k, minit='random')
            kmeans2(data[:, :1], k, minit='random')

    @skip_if_array_api_gpu
    @array_api_compatible
    @pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MemoryError in Wine.')
    def test_krandinit(self, xp):
        data = xp.asarray(TESTDATA_2D)
        datas = [xp.reshape(data, (200, 2)), xp.reshape(data, (20, 20))[:10, :]]
        k = int(1000000.0)
        for data in datas:
            rng = np.random.default_rng(1234)
            init = _krandinit(data, k, rng, xp)
            orig_cov = cov(data.T)
            init_cov = cov(init.T)
            xp_assert_close(orig_cov, init_cov, atol=0.01)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_empty(self, xp):
        assert_raises(ValueError, kmeans2, xp.asarray([]), xp.asarray(2))

    @skip_if_array_api
    def test_kmeans_0k(self):
        assert_raises(ValueError, kmeans, X, 0)
        assert_raises(ValueError, kmeans2, X, 0)
        assert_raises(ValueError, kmeans2, X, np.array([]))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans_large_thres(self, xp):
        x = xp.asarray([1, 2, 3, 4, 10], dtype=xp.float64)
        res = kmeans(x, xp.asarray(1), thresh=1e+16)
        xp_assert_close(res[0], xp.asarray([4.0], dtype=xp.float64))
        xp_assert_close(res[1], xp.asarray(2.4, dtype=xp.float64)[()])

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_kpp_low_dim(self, xp):
        prev_res = xp.asarray([[-1.95266667, 0.898], [-3.153375, 3.3945]], dtype=xp.float64)
        np.random.seed(42)
        res, _ = kmeans2(xp.asarray(TESTDATA_2D), xp.asarray(2), minit='++')
        xp_assert_close(res, prev_res)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans2_kpp_high_dim(self, xp):
        n_dim = 100
        size = 10
        centers = np.vstack([5 * np.ones(n_dim), -5 * np.ones(n_dim)])
        np.random.seed(42)
        data = np.vstack([np.random.multivariate_normal(centers[0], np.eye(n_dim), size=size), np.random.multivariate_normal(centers[1], np.eye(n_dim), size=size)])
        data = xp.asarray(data)
        res, _ = kmeans2(data, xp.asarray(2), minit='++')
        xp_assert_equal(xp.sign(res), xp.sign(xp.asarray(centers)))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_kmeans_diff_convergence(self, xp):
        obs = xp.asarray([-3, -1, 0, 1, 1, 8], dtype=xp.float64)
        res = kmeans(obs, xp.asarray([-3.0, 0.99]))
        xp_assert_close(res[0], xp.asarray([-0.4, 8.0], dtype=xp.float64))
        xp_assert_close(res[1], xp.asarray(1.0666666666666667, dtype=xp.float64)[()])

    @skip_if_array_api
    def test_kmeans_and_kmeans2_random_seed(self):
        seed_list = [1234, np.random.RandomState(1234), np.random.default_rng(1234)]
        for seed in seed_list:
            res1, _ = kmeans(TESTDATA_2D, 2, seed=seed)
            res2, _ = kmeans(TESTDATA_2D, 2, seed=seed)
            assert_allclose(res1, res1)
            for minit in ['random', 'points', '++']:
                res1, _ = kmeans2(TESTDATA_2D, 2, minit=minit, seed=seed)
                res2, _ = kmeans2(TESTDATA_2D, 2, minit=minit, seed=seed)
                assert_allclose(res1, res1)