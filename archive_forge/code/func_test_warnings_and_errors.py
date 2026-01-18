from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_warnings_and_errors(self):
    with warnings.catch_warnings(record=True) as w:
        pc = PCA(self.x, ncomp=300)
        assert_equal(len(w), 1)
    with warnings.catch_warnings(record=True) as w:
        rs = self.rs
        x = rs.standard_normal((200, 1)) * np.ones(200)
        pc = PCA(x, method='eig')
        assert_equal(len(w), 1)
    assert_raises(ValueError, PCA, self.x, method='unknown')
    assert_raises(ValueError, PCA, self.x, missing='unknown')
    assert_raises(ValueError, PCA, self.x, tol=2.0)
    assert_raises(ValueError, PCA, np.nan * np.ones((200, 100)), tol=2.0)