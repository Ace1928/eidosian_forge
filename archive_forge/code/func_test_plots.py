from statsmodels.compat.pandas import testing as pdt
import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import MLEInfluence
@pytest.mark.smoke
@pytest.mark.matplotlib
def test_plots(self, close_figures):
    infl1 = self.infl1
    infl0 = self.infl0
    fig = infl0.plot_influence(external=False)
    fig = infl1.plot_influence(external=False)
    fig = infl0.plot_index('resid', threshold=0.2, title='')
    fig = infl1.plot_index('resid', threshold=0.2, title='')
    fig = infl0.plot_index('dfbeta', idx=1, threshold=0.2, title='')
    fig = infl1.plot_index('dfbeta', idx=1, threshold=0.2, title='')
    fig = infl0.plot_index('cook', idx=1, threshold=0.2, title='')
    fig = infl1.plot_index('cook', idx=1, threshold=0.2, title='')
    fig = infl0.plot_index('hat', idx=1, threshold=0.2, title='')
    fig = infl1.plot_index('hat', idx=1, threshold=0.2, title='')