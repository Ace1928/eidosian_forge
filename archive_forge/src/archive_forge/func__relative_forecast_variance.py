from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
def _relative_forecast_variance(self, steps):
    """
        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
           principles and practice*, 3rd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
        """
    h = steps
    alpha = self.smoothing_level
    if self.has_trend:
        beta = self.smoothing_trend
    if self.has_seasonal:
        gamma = self.smoothing_seasonal
        m = self.seasonal_periods
        k = np.asarray((h - 1) / m, dtype=int)
    if self.damped_trend:
        phi = self.damping_trend
    model = self.model.short_name
    if model == 'ANN':
        return 1 + alpha ** 2 * (h - 1)
    elif model == 'AAN':
        return 1 + (h - 1) * (alpha ** 2 + alpha * beta * h + beta ** 2 * h / 6 * (2 * h - 1))
    elif model == 'AAdN':
        return 1 + alpha ** 2 * (h - 1) + beta * phi * h / (1 - phi) ** 2 * (2 * alpha * (1 - phi) + beta * phi) - beta * phi * (1 - phi ** h) / ((1 - phi) ** 2 * (1 - phi ** 2)) * (2 * alpha * (1 - phi ** 2) + beta * phi * (1 + 2 * phi - phi ** h))
    elif model == 'ANA':
        return 1 + alpha ** 2 * (h - 1) + gamma * k * (2 * alpha + gamma)
    elif model == 'AAA':
        return 1 + (h - 1) * (alpha ** 2 + alpha * beta * h + beta ** 2 / 6 * h * (2 * h - 1)) + gamma * k * (2 * alpha + gamma + beta * m * (k + 1))
    elif model == 'AAdA':
        return 1 + alpha ** 2 * (h - 1) + gamma * k * (2 * alpha + gamma) + beta * phi * h / (1 - phi) ** 2 * (2 * alpha * (1 - phi) + beta * phi) - beta * phi * (1 - phi ** h) / ((1 - phi) ** 2 * (1 - phi ** 2)) * (2 * alpha * (1 - phi ** 2) + beta * phi * (1 + 2 * phi - phi ** h)) + 2 * beta * gamma * phi / ((1 - phi) * (1 - phi ** m)) * (k * (1 - phi ** m) - phi ** m * (1 - phi ** (m * k)))
    else:
        raise NotImplementedError