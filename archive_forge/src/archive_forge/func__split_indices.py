import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def _split_indices(self, vec):
    null = pd.isnull(vec)
    ix_obs = np.flatnonzero(~null)
    ix_miss = np.flatnonzero(null)
    if len(ix_obs) == 0:
        raise ValueError('variable to be imputed has no observed values')
    return (ix_obs, ix_miss)