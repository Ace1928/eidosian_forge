import numpy as np
from numpy.testing import assert_array_almost_equal
from statsmodels.sandbox.tools import pca, pcasvd
from statsmodels.multivariate.tests.results.datamlw import (
def test_pca_svd():
    xreduced, factors, evals, evecs = pca(xf)
    factors_wconst = np.c_[factors, np.ones((factors.shape[0], 1))]
    beta = np.dot(np.linalg.pinv(factors_wconst), xf)
    assert_array_almost_equal(beta.T[:, :4], evecs, 14)
    xred_svd, factors_svd, evals_svd, evecs_svd = pcasvd(xf, keepdim=0)
    assert_array_almost_equal(evals_svd, evals, 14)
    msign = (evecs / evecs_svd)[0]
    assert_array_almost_equal(msign * evecs_svd, evecs, 13)
    assert_array_almost_equal(msign * factors_svd, factors, 12)
    assert_array_almost_equal(xred_svd, xreduced, 13)
    pcares = pca(xf, keepdim=2)
    pcasvdres = pcasvd(xf, keepdim=2)
    check_pca_svd(pcares, pcasvdres)