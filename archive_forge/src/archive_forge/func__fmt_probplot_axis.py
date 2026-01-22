from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
def _fmt_probplot_axis(ax, dist, nobs):
    """
    Formats a theoretical quantile axis to display the corresponding
    probabilities on the quantiles' scale.

    Parameters
    ----------
    ax : AxesSubplot, optional
        The axis to be formatted
    nobs : scalar
        Number of observations in the sample
    dist : scipy.stats.distribution
        A scipy.stats distribution sufficiently specified to implement its
        ppf() method.

    Returns
    -------
    There is no return value. This operates on `ax` in place
    """
    _check_for(dist, 'ppf')
    axis_probs = np.linspace(10, 90, 9, dtype=float)
    small = np.array([1.0, 2, 5])
    axis_probs = np.r_[small, axis_probs, 100 - small[::-1]]
    if nobs >= 50:
        axis_probs = np.r_[small / 10, axis_probs, 100 - small[::-1] / 10]
    if nobs >= 500:
        axis_probs = np.r_[small / 100, axis_probs, 100 - small[::-1] / 100]
    axis_probs /= 100.0
    axis_qntls = dist.ppf(axis_probs)
    ax.set_xticks(axis_qntls)
    ax.set_xticklabels([str(lbl) for lbl in axis_probs * 100], rotation=45, rotation_mode='anchor', horizontalalignment='right', verticalalignment='center')
    ax.set_xlim([axis_qntls.min(), axis_qntls.max()])