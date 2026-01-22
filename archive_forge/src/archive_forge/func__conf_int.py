from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import (
from collections import namedtuple
import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats
from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import (
from statsmodels.tools.validation import array_like, int_like, string_like
def _conf_int(self, alpha, cols):
    bse = np.asarray(self.bse)
    if self.use_t:
        dist = stats.t
        df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        df_resid = np.asarray(df_resid)[:, None]
        q = dist.ppf(1 - alpha / 2, df_resid)
    else:
        dist = stats.norm
        q = dist.ppf(1 - alpha / 2)
    params = np.asarray(self.params)
    lower = params - q * bse
    upper = params + q * bse
    if cols is not None:
        cols = np.asarray(cols)
        lower = lower[:, cols]
        upper = upper[:, cols]
    return np.asarray(list(zip(lower, upper)))