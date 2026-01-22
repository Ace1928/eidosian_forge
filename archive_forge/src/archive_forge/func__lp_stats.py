from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def _lp_stats(self, fep_mean, fep_sd, vc_mean, vc_sd):
    tm = np.dot(self.exog, fep_mean)
    tv = np.dot(self.exog ** 2, fep_sd ** 2)
    tm += self.exog_vc.dot(vc_mean)
    tv += self.exog_vc2.dot(vc_sd ** 2)
    return (tm, tv)