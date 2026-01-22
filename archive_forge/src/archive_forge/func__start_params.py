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
@property
def _start_params(self):
    """
        Default start params in the format of external parameters.
        This should not be called directly, but by calling
        ``self.start_params``.
        """
    params = []
    for p in self._smoothing_param_names:
        if p in self.param_names:
            params.append(self._default_start_params[p])
    if self.initialization_method == 'estimated':
        lvl_idx = len(params)
        params += [self.initial_level]
        if self.has_trend:
            params += [self.initial_trend]
        if self.has_seasonal:
            initial_seasonal = self.initial_seasonal
            if self.seasonal == 'mul':
                params[lvl_idx] *= initial_seasonal[-1]
                initial_seasonal /= initial_seasonal[-1]
            else:
                params[lvl_idx] += initial_seasonal[-1]
                initial_seasonal -= initial_seasonal[-1]
            params += initial_seasonal.tolist()
    return np.array(params)