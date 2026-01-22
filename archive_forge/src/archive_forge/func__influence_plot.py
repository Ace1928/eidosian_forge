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
@Appender(_plot_influence_doc.format(**{'extra_params_doc': 'results: object\n        Results for a fitted regression model.\n    influence: instance\n        The instance of Influence for model.'}))
def _influence_plot(results, influence, external=True, alpha=0.05, criterion='cooks', size=48, plot_alpha=0.75, ax=None, leverage=None, resid=None, **kwargs):
    infl = influence
    fig, ax = utils.create_mpl_ax(ax)
    if criterion.lower().startswith('coo'):
        psize = infl.cooks_distance[0]
    elif criterion.lower().startswith('dff'):
        psize = np.abs(infl.dffits[0])
    else:
        raise ValueError('Criterion %s not understood' % criterion)
    old_range = np.ptp(psize)
    new_range = size ** 2 - 8 ** 2
    psize = (psize - psize.min()) * new_range / old_range + 8 ** 2
    if leverage is None:
        leverage = infl.hat_matrix_diag
    if resid is None:
        ylabel = 'Studentized Residuals'
        if external:
            resid = infl.resid_studentized_external
        else:
            resid = infl.resid_studentized
    else:
        resid = np.asarray(resid)
        ylabel = 'Residuals'
    from scipy import stats
    cutoff = stats.t.ppf(1.0 - alpha / 2, results.df_resid)
    large_resid = np.abs(resid) > cutoff
    large_leverage = leverage > _high_leverage(results)
    large_points = np.logical_or(large_resid, large_leverage)
    ax.scatter(leverage, resid, s=psize, alpha=plot_alpha)
    labels = results.model.data.row_labels
    if labels is None:
        labels = lrange(len(resid))
    ax = utils.annotate_axes(np.where(large_points)[0], labels, lzip(leverage, resid), lzip(-(psize / 2) ** 0.5, (psize / 2) ** 0.5), 'x-large', ax)
    font = {'fontsize': 16, 'color': 'black'}
    ax.set_ylabel(ylabel, **font)
    ax.set_xlabel('Leverage', **font)
    ax.set_title('Influence Plot', **font)
    return fig