import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
class TestPdist:

    def setup_method(self):
        self.rnd_eo_names = ['random-float32-data', 'random-int-data', 'random-uint-data', 'random-double-data', 'random-bool-data']
        self.valid_upcasts = {'bool': [np_ulong, np_long, np.float32, np.float64], 'uint': [np_long, np.float32, np.float64], 'int': [np.float32, np.float64], 'float32': [np.float64]}

    def test_pdist_extra_args(self, metric):
        X1 = [[1.0, 2.0], [1.2, 2.3], [2.2, 2.3]]
        kwargs = {'N0tV4l1D_p4raM': 3.14, 'w': np.arange(2)}
        args = [3.14] * 200
        with pytest.raises(TypeError):
            pdist(X1, metric=metric, **kwargs)
        with pytest.raises(TypeError):
            pdist(X1, metric=eval(metric), **kwargs)
        with pytest.raises(TypeError):
            pdist(X1, metric='test_' + metric, **kwargs)
        with pytest.raises(TypeError):
            pdist(X1, *args, metric=metric)
        with pytest.raises(TypeError):
            pdist(X1, *args, metric=eval(metric))
        with pytest.raises(TypeError):
            pdist(X1, *args, metric='test_' + metric)

    def test_pdist_extra_args_custom(self):

        def _my_metric(x, y, arg, kwarg=1, kwarg2=2):
            return arg + kwarg + kwarg2
        X1 = [[1.0, 2.0], [1.2, 2.3], [2.2, 2.3]]
        kwargs = {'N0tV4l1D_p4raM': 3.14, 'w': np.arange(2)}
        args = [3.14] * 200
        with pytest.raises(TypeError):
            pdist(X1, _my_metric)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, *args)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, **kwargs)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, kwarg=2.2, kwarg2=3.3)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, 1, 2, kwarg=2.2)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, 1, 2, kwarg=2.2)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, 1.1, 2.2, 3.3)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, 1.1, 2.2)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, 1.1)
        with pytest.raises(TypeError):
            pdist(X1, _my_metric, 1.1, kwarg=2.2, kwarg2=3.3)
        assert_allclose(pdist(X1, metric=_my_metric, arg=1.1, kwarg2=3.3), 5.4)

    def test_pdist_euclidean_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_euclidean_random_u(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_euclidean_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-euclidean']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_euclidean_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        Y_test2 = wpdist_no_const(X, 'test_euclidean')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_euclidean_iris_double(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-euclidean-iris']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_euclidean_iris_float32(self):
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-euclidean-iris']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    @pytest.mark.slow
    def test_pdist_euclidean_iris_nonC(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-euclidean-iris']
        Y_test2 = wpdist_no_const(X, 'test_euclidean')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_seuclidean_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-seuclidean']
        Y_test1 = pdist(X, 'seuclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_seuclidean_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-seuclidean']
        Y_test1 = pdist(X, 'seuclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)
        V = np.var(X, axis=0, ddof=1)
        Y_test2 = pdist(X, 'seuclidean', V=V)
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_seuclidean_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-seuclidean']
        Y_test2 = pdist(X, 'test_seuclidean')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_seuclidean_iris(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-seuclidean-iris']
        Y_test1 = pdist(X, 'seuclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_seuclidean_iris_float32(self):
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-seuclidean-iris']
        Y_test1 = pdist(X, 'seuclidean')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_seuclidean_iris_nonC(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-seuclidean-iris']
        Y_test2 = pdist(X, 'test_seuclidean')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_cosine_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cosine']
        Y_test1 = wpdist(X, 'cosine')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_cosine_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-cosine']
        Y_test1 = wpdist(X, 'cosine')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_cosine_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cosine']
        Y_test2 = wpdist(X, 'test_cosine')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_cosine_iris(self):
        eps = 1e-05
        X = eo['iris']
        Y_right = eo['pdist-cosine-iris']
        Y_test1 = wpdist(X, 'cosine')
        assert_allclose(Y_test1, Y_right, atol=eps)

    @pytest.mark.slow
    def test_pdist_cosine_iris_float32(self):
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-cosine-iris']
        Y_test1 = wpdist(X, 'cosine')
        assert_allclose(Y_test1, Y_right, atol=eps, verbose=verbose > 2)

    @pytest.mark.slow
    def test_pdist_cosine_iris_nonC(self):
        eps = 1e-05
        X = eo['iris']
        Y_right = eo['pdist-cosine-iris']
        Y_test2 = wpdist(X, 'test_cosine')
        assert_allclose(Y_test2, Y_right, atol=eps)

    def test_pdist_cosine_bounds(self):
        x = np.abs(np.random.RandomState(1337).rand(91))
        X = np.vstack([x, x])
        assert_(wpdist(X, 'cosine')[0] >= 0, msg='cosine distance should be non-negative')

    def test_pdist_cityblock_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cityblock']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_cityblock_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-cityblock']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_cityblock_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cityblock']
        Y_test2 = wpdist_no_const(X, 'test_cityblock')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_cityblock_iris(self):
        eps = 1e-14
        X = eo['iris']
        Y_right = eo['pdist-cityblock-iris']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_cityblock_iris_float32(self):
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-cityblock-iris']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    @pytest.mark.slow
    def test_pdist_cityblock_iris_nonC(self):
        eps = 1e-14
        X = eo['iris']
        Y_right = eo['pdist-cityblock-iris']
        Y_test2 = wpdist_no_const(X, 'test_cityblock')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_correlation_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-correlation']
        Y_test1 = wpdist(X, 'correlation')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_correlation_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-correlation']
        Y_test1 = wpdist(X, 'correlation')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_correlation_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-correlation']
        Y_test2 = wpdist(X, 'test_correlation')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_correlation_iris(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-correlation-iris']
        Y_test1 = wpdist(X, 'correlation')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_correlation_iris_float32(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = np.float32(eo['pdist-correlation-iris'])
        Y_test1 = wpdist(X, 'correlation')
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    @pytest.mark.slow
    def test_pdist_correlation_iris_nonC(self):
        if sys.maxsize > 2 ** 32:
            eps = 1e-07
        else:
            pytest.skip('see gh-16456')
        X = eo['iris']
        Y_right = eo['pdist-correlation-iris']
        Y_test2 = wpdist(X, 'test_correlation')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.parametrize('p', [0.1, 0.25, 1.0, 2.0, 3.2, np.inf])
    def test_pdist_minkowski_random_p(self, p):
        eps = 1e-13
        X = eo['pdist-double-inp']
        Y1 = wpdist_no_const(X, 'minkowski', p=p)
        Y2 = wpdist_no_const(X, 'test_minkowski', p=p)
        assert_allclose(Y1, Y2, atol=0, rtol=eps)

    def test_pdist_minkowski_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-minkowski-3.2']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_minkowski_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-minkowski-3.2']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_minkowski_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-minkowski-3.2']
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=3.2)
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-3.2-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris_float32(self):
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-minkowski-3.2-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris_nonC(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-3.2-iris']
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=3.2)
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-5.8-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=5.8)
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris_float32(self):
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-minkowski-5.8-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=5.8)
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris_nonC(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-5.8-iris']
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=5.8)
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_mahalanobis(self):
        x = np.array([2.0, 2.0, 3.0, 5.0]).reshape(-1, 1)
        dist = pdist(x, metric='mahalanobis')
        assert_allclose(dist, [0.0, np.sqrt(0.5), np.sqrt(4.5), np.sqrt(0.5), np.sqrt(4.5), np.sqrt(2.0)])
        x = np.array([[0, 0], [-1, 0], [0, 2], [1, 0], [0, -2]])
        dist = pdist(x, metric='mahalanobis')
        rt2 = np.sqrt(2)
        assert_allclose(dist, [rt2, rt2, rt2, rt2, 2, 2 * rt2, 2, 2, 2 * rt2, 2])
        with pytest.raises(ValueError):
            wpdist([[0, 1], [2, 3]], metric='mahalanobis')

    def test_pdist_hamming_random(self):
        eps = 1e-15
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_hamming_random_float32(self):
        eps = 1e-15
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_hamming_random_nonC(self):
        eps = 1e-15
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-hamming']
        Y_test2 = wpdist(X, 'test_hamming')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_dhamming_random(self):
        eps = 1e-15
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_dhamming_random_float32(self):
        eps = 1e-15
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_dhamming_random_nonC(self):
        eps = 1e-15
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test2 = wpdist(X, 'test_hamming')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_jaccard_random(self):
        eps = 1e-08
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_jaccard_random_float32(self):
        eps = 1e-08
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_jaccard_random_nonC(self):
        eps = 1e-08
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-jaccard']
        Y_test2 = wpdist(X, 'test_jaccard')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_djaccard_random(self):
        eps = 1e-08
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_djaccard_random_float32(self):
        eps = 1e-08
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_djaccard_allzeros(self):
        eps = 1e-15
        Y = pdist(np.zeros((5, 3)), 'jaccard')
        assert_allclose(np.zeros(10), Y, rtol=eps)

    def test_pdist_djaccard_random_nonC(self):
        eps = 1e-08
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test2 = wpdist(X, 'test_jaccard')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_jensenshannon_random(self):
        eps = 1e-11
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-jensenshannon']
        Y_test1 = pdist(X, 'jensenshannon')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_jensenshannon_random_float32(self):
        eps = 1e-08
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-jensenshannon']
        Y_test1 = pdist(X, 'jensenshannon')
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    def test_pdist_jensenshannon_random_nonC(self):
        eps = 1e-11
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-jensenshannon']
        Y_test2 = pdist(X, 'test_jensenshannon')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_jensenshannon_iris(self):
        if _is_32bit():
            eps = 2.5e-10
        else:
            eps = 1e-12
        X = eo['iris']
        Y_right = eo['pdist-jensenshannon-iris']
        Y_test1 = pdist(X, 'jensenshannon')
        assert_allclose(Y_test1, Y_right, atol=eps)

    def test_pdist_jensenshannon_iris_float32(self):
        eps = 1e-06
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-jensenshannon-iris']
        Y_test1 = pdist(X, 'jensenshannon')
        assert_allclose(Y_test1, Y_right, atol=eps, verbose=verbose > 2)

    def test_pdist_jensenshannon_iris_nonC(self):
        eps = 5e-05
        X = eo['iris']
        Y_right = eo['pdist-jensenshannon-iris']
        Y_test2 = pdist(X, 'test_jensenshannon')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_djaccard_allzeros_nonC(self):
        eps = 1e-15
        Y = pdist(np.zeros((5, 3)), 'test_jaccard')
        assert_allclose(np.zeros(10), Y, rtol=eps)

    def test_pdist_chebyshev_random(self):
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-chebyshev']
        Y_test1 = pdist(X, 'chebyshev')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_chebyshev_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-chebyshev']
        Y_test1 = pdist(X, 'chebyshev')
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    def test_pdist_chebyshev_random_nonC(self):
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-chebyshev']
        Y_test2 = pdist(X, 'test_chebyshev')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_chebyshev_iris(self):
        eps = 1e-14
        X = eo['iris']
        Y_right = eo['pdist-chebyshev-iris']
        Y_test1 = pdist(X, 'chebyshev')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_chebyshev_iris_float32(self):
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-chebyshev-iris']
        Y_test1 = pdist(X, 'chebyshev')
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    def test_pdist_chebyshev_iris_nonC(self):
        eps = 1e-14
        X = eo['iris']
        Y_right = eo['pdist-chebyshev-iris']
        Y_test2 = pdist(X, 'test_chebyshev')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    def test_pdist_matching_mtica1(self):
        m = wmatching(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
        m2 = wmatching(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
        assert_allclose(m, 0.6, rtol=0, atol=1e-10)
        assert_allclose(m2, 0.6, rtol=0, atol=1e-10)

    def test_pdist_matching_mtica2(self):
        m = wmatching(np.array([1, 0, 1]), np.array([1, 1, 0]))
        m2 = wmatching(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool))
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)

    def test_pdist_jaccard_mtica1(self):
        m = wjaccard(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
        m2 = wjaccard(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
        assert_allclose(m, 0.6, rtol=0, atol=1e-10)
        assert_allclose(m2, 0.6, rtol=0, atol=1e-10)

    def test_pdist_jaccard_mtica2(self):
        m = wjaccard(np.array([1, 0, 1]), np.array([1, 1, 0]))
        m2 = wjaccard(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool))
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)

    def test_pdist_yule_mtica1(self):
        m = wyule(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
        m2 = wyule(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 2, rtol=0, atol=1e-10)
        assert_allclose(m2, 2, rtol=0, atol=1e-10)

    def test_pdist_yule_mtica2(self):
        m = wyule(np.array([1, 0, 1]), np.array([1, 1, 0]))
        m2 = wyule(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 2, rtol=0, atol=1e-10)
        assert_allclose(m2, 2, rtol=0, atol=1e-10)

    def test_pdist_dice_mtica1(self):
        m = wdice(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
        m2 = wdice(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 7, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 7, rtol=0, atol=1e-10)

    def test_pdist_dice_mtica2(self):
        m = wdice(np.array([1, 0, 1]), np.array([1, 1, 0]))
        m2 = wdice(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 0.5, rtol=0, atol=1e-10)
        assert_allclose(m2, 0.5, rtol=0, atol=1e-10)

    def test_pdist_sokalsneath_mtica1(self):
        m = sokalsneath(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
        m2 = sokalsneath(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 4, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 4, rtol=0, atol=1e-10)

    def test_pdist_sokalsneath_mtica2(self):
        m = wsokalsneath(np.array([1, 0, 1]), np.array([1, 1, 0]))
        m2 = wsokalsneath(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 4 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 4 / 5, rtol=0, atol=1e-10)

    def test_pdist_rogerstanimoto_mtica1(self):
        m = wrogerstanimoto(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
        m2 = wrogerstanimoto(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 4, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 4, rtol=0, atol=1e-10)

    def test_pdist_rogerstanimoto_mtica2(self):
        m = wrogerstanimoto(np.array([1, 0, 1]), np.array([1, 1, 0]))
        m2 = wrogerstanimoto(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 4 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 4 / 5, rtol=0, atol=1e-10)

    def test_pdist_russellrao_mtica1(self):
        m = wrussellrao(np.array([1, 0, 1, 1, 0]), np.array([1, 1, 0, 1, 1]))
        m2 = wrussellrao(np.array([1, 0, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 5, rtol=0, atol=1e-10)

    def test_pdist_russellrao_mtica2(self):
        m = wrussellrao(np.array([1, 0, 1]), np.array([1, 1, 0]))
        m2 = wrussellrao(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)

    @pytest.mark.slow
    def test_pdist_canberra_match(self):
        D = eo['iris']
        if verbose > 2:
            print(D.shape, D.dtype)
        eps = 1e-15
        y1 = wpdist_no_const(D, 'canberra')
        y2 = wpdist_no_const(D, 'test_canberra')
        assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)

    def test_pdist_canberra_ticket_711(self):
        eps = 1e-08
        pdist_y = wpdist_no_const(([3.3], [3.4]), 'canberra')
        right_y = 0.01492537
        assert_allclose(pdist_y, right_y, atol=eps, verbose=verbose > 2)

    def test_pdist_custom_notdouble(self):

        class myclass:
            pass

        def _my_metric(x, y):
            if not isinstance(x[0], myclass) or not isinstance(y[0], myclass):
                raise ValueError('Type has been changed')
            return 1.123
        data = np.array([[myclass()], [myclass()]], dtype=object)
        pdist_y = pdist(data, metric=_my_metric)
        right_y = 1.123
        assert_equal(pdist_y, right_y, verbose=verbose > 2)

    def _check_calling_conventions(self, X, metric, eps=1e-07, **kwargs):
        try:
            y1 = pdist(X, metric=metric, **kwargs)
            y2 = pdist(X, metric=eval(metric), **kwargs)
            y3 = pdist(X, metric='test_' + metric, **kwargs)
        except Exception as e:
            e_cls = e.__class__
            if verbose > 2:
                print(e_cls.__name__)
                print(e)
            with pytest.raises(e_cls):
                pdist(X, metric=metric, **kwargs)
            with pytest.raises(e_cls):
                pdist(X, metric=eval(metric), **kwargs)
            with pytest.raises(e_cls):
                pdist(X, metric='test_' + metric, **kwargs)
        else:
            assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)
            assert_allclose(y1, y3, rtol=eps, verbose=verbose > 2)

    def test_pdist_calling_conventions(self, metric):
        for eo_name in self.rnd_eo_names:
            X = eo[eo_name][::5, ::2]
            if verbose > 2:
                print('testing: ', metric, ' with: ', eo_name)
            if metric in {'dice', 'yule', 'matching', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'kulczynski1'} and 'bool' not in eo_name:
                continue
            self._check_calling_conventions(X, metric)
            if metric == 'seuclidean':
                V = np.var(X.astype(np.float64), axis=0, ddof=1)
                self._check_calling_conventions(X, metric, V=V)
            elif metric == 'mahalanobis':
                V = np.atleast_2d(np.cov(X.astype(np.float64).T))
                VI = np.array(np.linalg.inv(V).T)
                self._check_calling_conventions(X, metric, VI=VI)

    def test_pdist_dtype_equivalence(self, metric):
        eps = 1e-07
        tests = [(eo['random-bool-data'], self.valid_upcasts['bool']), (eo['random-uint-data'], self.valid_upcasts['uint']), (eo['random-int-data'], self.valid_upcasts['int']), (eo['random-float32-data'], self.valid_upcasts['float32'])]
        for test in tests:
            X1 = test[0][::5, ::2]
            try:
                y1 = pdist(X1, metric=metric)
            except Exception as e:
                e_cls = e.__class__
                if verbose > 2:
                    print(e_cls.__name__)
                    print(e)
                for new_type in test[1]:
                    X2 = new_type(X1)
                    with pytest.raises(e_cls):
                        pdist(X2, metric=metric)
            else:
                for new_type in test[1]:
                    y2 = pdist(new_type(X1), metric=metric)
                    assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)

    def test_pdist_out(self, metric):
        eps = 1e-15
        X = eo['random-float32-data'][::5, ::2]
        out_size = int(X.shape[0] * (X.shape[0] - 1) / 2)
        kwargs = dict()
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        out1 = np.empty(out_size, dtype=np.float64)
        Y_right = pdist(X, metric, **kwargs)
        Y_test1 = pdist(X, metric, out=out1, **kwargs)
        assert_allclose(Y_test1, Y_right, rtol=eps)
        assert_(Y_test1 is out1)
        out2 = np.empty(out_size + 3, dtype=np.float64)
        with pytest.raises(ValueError):
            pdist(X, metric, out=out2, **kwargs)
        out3 = np.empty(2 * out_size, dtype=np.float64)[::2]
        with pytest.raises(ValueError):
            pdist(X, metric, out=out3, **kwargs)
        out5 = np.empty(out_size, dtype=np.int64)
        with pytest.raises(ValueError):
            pdist(X, metric, out=out5, **kwargs)

    def test_striding(self, metric):
        eps = 1e-15
        X = eo['random-float32-data'][::5, ::2]
        X_copy = X.copy()
        assert_(not X.flags.c_contiguous)
        assert_(X_copy.flags.c_contiguous)
        kwargs = dict()
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        Y1 = pdist(X, metric, **kwargs)
        Y2 = pdist(X_copy, metric, **kwargs)
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)