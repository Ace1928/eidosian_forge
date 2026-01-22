import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def _boot_kwds(self, kwds, rix):
    for k in kwds:
        v = kwds[k]
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim == 1 and v.shape[0] == len(rix):
            kwds[k] = v[rix]
        if v.ndim == 2 and v.shape[0] == len(rix):
            kwds[k] = v[rix, :]
    return kwds