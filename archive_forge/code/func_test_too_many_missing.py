from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_too_many_missing(reset_randomstate):
    data = np.random.standard_normal((200, 50))
    data[0, :-3] = np.nan
    with pytest.raises(ValueError):
        PCA(data, ncomp=5, missing='drop-col')
    p = PCA(data, missing='drop-min')
    assert max(p.factors.shape) == max(data.shape) - 1