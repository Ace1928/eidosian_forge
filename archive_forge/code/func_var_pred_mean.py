import numpy as np
import pandas as pd
from scipy import stats
@property
def var_pred_mean(self):
    """The variance of the predicted mean"""
    if self._var_pred_mean.ndim > 2:
        return self._var_pred_mean
    return self._wrap_pandas(self._var_pred_mean, 'var_pred_mean')