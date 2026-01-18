import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_whitening():
    X = datasets.make_low_rank_matrix(1000, 10, tail_strength=0.0, effective_rank=2, random_state=1999)
    prec = 3
    n_samples, n_features = X.shape
    for nc in [None, 9]:
        pca = PCA(whiten=True, n_components=nc).fit(X)
        ipca = IncrementalPCA(whiten=True, n_components=nc, batch_size=250).fit(X)
        Xt_pca = pca.transform(X)
        Xt_ipca = ipca.transform(X)
        assert_almost_equal(np.abs(Xt_pca), np.abs(Xt_ipca), decimal=prec)
        Xinv_ipca = ipca.inverse_transform(Xt_ipca)
        Xinv_pca = pca.inverse_transform(Xt_pca)
        assert_almost_equal(X, Xinv_ipca, decimal=prec)
        assert_almost_equal(X, Xinv_pca, decimal=prec)
        assert_almost_equal(Xinv_pca, Xinv_ipca, decimal=prec)