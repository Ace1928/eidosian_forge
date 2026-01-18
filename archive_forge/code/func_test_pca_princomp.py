import numpy as np
from numpy.testing import assert_array_almost_equal
from statsmodels.sandbox.tools import pca, pcasvd
from statsmodels.multivariate.tests.results.datamlw import (
def test_pca_princomp():
    pcares = pca(xf)
    check_pca_princomp(pcares, princomp1)
    pcares = pca(xf[:20, :])
    check_pca_princomp(pcares, princomp2)
    pcares = pca(xf[:20, :] - xf[:20, :].mean(0))
    check_pca_princomp(pcares, princomp3)
    pcares = pca(xf[:20, :] - xf[:20, :].mean(0), demean=0)
    check_pca_princomp(pcares, princomp3)