from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import GLS, OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tools.tools import maybe_unwrap_results
from ._regressionplots_doc import (
@Appender(_plot_leverage_resid2_doc.format({'extra_params_doc': 'results: object\n    Results for a fitted regression model\ninfluence: instance\n    instance of Influence for model'}))
def _plot_leverage_resid2(results, influence, alpha=0.05, ax=None, **kwargs):
    from scipy.stats import norm, zscore
    fig, ax = utils.create_mpl_ax(ax)
    infl = influence
    leverage = infl.hat_matrix_diag
    resid = zscore(infl.resid)
    ax.plot(resid ** 2, leverage, 'o', **kwargs)
    ax.set_xlabel('Normalized residuals**2')
    ax.set_ylabel('Leverage')
    ax.set_title('Leverage vs. Normalized residuals squared')
    large_leverage = leverage > _high_leverage(results)
    cutoff = norm.ppf(1.0 - alpha / 2)
    large_resid = np.abs(resid) > cutoff
    labels = results.model.data.row_labels
    if labels is None:
        labels = lrange(int(results.nobs))
    index = np.where(np.logical_or(large_leverage, large_resid))[0]
    ax = utils.annotate_axes(index, labels, lzip(resid ** 2, leverage), [(0, 5)] * int(results.nobs), 'large', ax=ax, ha='center', va='bottom')
    ax.margins(0.075, 0.075)
    return fig