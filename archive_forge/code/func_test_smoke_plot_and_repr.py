from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
@pytest.mark.smoke
@pytest.mark.matplotlib
def test_smoke_plot_and_repr(self, close_figures):
    pc = PCA(self.x)
    fig = pc.plot_scree()
    fig = pc.plot_scree(ncomp=10)
    fig = pc.plot_scree(log_scale=False)
    fig = pc.plot_scree(cumulative=True)
    fig = pc.plot_rsquare()
    fig = pc.plot_rsquare(ncomp=5)
    pc.__repr__()
    pc = PCA(self.x, standardize=False)
    pc.__repr__()
    pc = PCA(self.x, standardize=False, demean=False)
    pc.__repr__()
    pc = PCA(self.x, ncomp=2, gls=True)
    assert 'GLS' in pc.__repr__()
    assert_equal(self.x, pc.data)