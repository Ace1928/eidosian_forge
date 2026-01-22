import numpy as np
import pandas as pd
from statsmodels.graphics.utils import maybe_name_or_idx
def _simulate_params(self, result):
    """
        Simulate model parameters from fitted sampling distribution.
        """
    mn = result.params
    cov = result.cov_params()
    return np.random.multivariate_normal(mn, cov)