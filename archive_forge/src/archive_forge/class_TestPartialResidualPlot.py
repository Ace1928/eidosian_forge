import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
class TestPartialResidualPlot:

    @pytest.mark.matplotlib
    def test_partial_residual_poisson(self, close_figures):
        np.random.seed(3446)
        n = 100
        p = 3
        exog = np.random.normal(size=(n, p))
        exog[:, 0] = 1
        lin_pred = 4 + exog[:, 1] + 0.2 * exog[:, 2] ** 2
        expval = np.exp(lin_pred)
        endog = np.random.poisson(expval)
        model = sm.GLM(endog, exog, family=sm.families.Poisson())
        results = model.fit()
        for focus_col in (1, 2):
            for j in (0, 1):
                if j == 0:
                    fig = plot_partial_residuals(results, focus_col)
                else:
                    fig = results.plot_partial_residuals(focus_col)
                ax = fig.get_axes()[0]
                add_lowess(ax)
                ax.set_position([0.1, 0.1, 0.8, 0.77])
                effect_str = ['Intercept', 'Linear effect, slope=1', 'Quadratic effect'][focus_col]
                ti = 'Partial residual plot'
                if j == 1:
                    ti += ' (called as method)'
                ax.set_title(ti + '\nPoisson regression\n' + effect_str)
                close_or_save(pdf, fig)