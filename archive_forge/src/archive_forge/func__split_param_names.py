import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
def _split_param_names(self):
    xnames = self.data.param_names
    q = 0
    mean_names = xnames[q:q + self.k_exog]
    q += self.k_exog
    scale_names = xnames[q:q + self.k_scale]
    q += self.k_scale
    smooth_names = xnames[q:q + self.k_smooth]
    if self._has_noise:
        q += self.k_noise
        noise_names = xnames[q:q + self.k_noise]
    else:
        noise_names = []
    return (mean_names, scale_names, smooth_names, noise_names)