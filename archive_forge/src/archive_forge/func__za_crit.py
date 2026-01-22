from __future__ import annotations
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import lzip
from statsmodels.compat.scipy import _next_regular
from typing import Literal, Union
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
def _za_crit(self, stat, model='c'):
    """
        Linear interpolation for Zivot-Andrews p-values and critical values

        Parameters
        ----------
        stat : float
            The ZA test statistic
        model : {"c","t","ct"}
            The model used when computing the ZA statistic. "c" is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated ZA test statistic distribution
        """
    table = self._za_critical_values[model]
    pcnts = table[:, 0]
    stats = table[:, 1]
    pvalue = np.interp(stat, stats, pcnts) / 100.0
    cv = [1.0, 5.0, 10.0]
    crit_value = np.interp(cv, pcnts, stats)
    cvdict = {'1%': crit_value[0], '5%': crit_value[1], '10%': crit_value[2]}
    return (pvalue, cvdict)