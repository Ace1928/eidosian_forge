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
def _internal_params(self, params):
    """
        Converts a parameter array passed from outside to the internally used
        full parameter array.
        """
    internal = np.zeros(self._k_params_internal, dtype=params.dtype)
    for i, name in enumerate(self.param_names):
        internal_idx = self._internal_params_index[name]
        internal[internal_idx] = params[i]
    if not self.damped_trend:
        internal[3] = 1
    if self.initialization_method != 'estimated':
        internal[4] = self.initial_level
        internal[5] = self.initial_trend
        if np.isscalar(self.initial_seasonal):
            internal[6:] = self.initial_seasonal
        else:
            internal[6:] = self.initial_seasonal[::-1]
    return internal