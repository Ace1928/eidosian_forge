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
def add_lowess(ax, lines_idx=0, frac=0.2, **lowess_kwargs):
    """
    Add Lowess line to a plot.

    Parameters
    ----------
    ax : AxesSubplot
        The Axes to which to add the plot
    lines_idx : int
        This is the line on the existing plot to which you want to add
        a smoothed lowess line.
    frac : float
        The fraction of the points to use when doing the lowess fit.
    lowess_kwargs
        Additional keyword arguments are passes to lowess.

    Returns
    -------
    Figure
        The figure that holds the instance.
    """
    y0 = ax.get_lines()[lines_idx]._y
    x0 = ax.get_lines()[lines_idx]._x
    lres = lowess(y0, x0, frac=frac, **lowess_kwargs)
    ax.plot(lres[:, 0], lres[:, 1], 'r', lw=1.5)
    return ax.figure