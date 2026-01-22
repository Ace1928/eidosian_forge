import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
def _km_w_ties(self, tie_indic, untied_km):
    """
        Computes KM estimator value at each observation, taking into acocunt
        ties in the data.

        Parameters
        ----------
        tie_indic: 1d array
            Indicates if the i'th observation is the same as the ith +1
        untied_km: 1d array
            Km estimates at each observation assuming no ties.
        """
    num_same = 1
    idx_nums = []
    for obs_num in np.arange(int(self.nobs - 1))[::-1]:
        if tie_indic[obs_num] == 1:
            idx_nums.append(obs_num)
            num_same = num_same + 1
            untied_km[obs_num] = untied_km[obs_num + 1]
        elif tie_indic[obs_num] == 0 and num_same > 1:
            idx_nums.append(max(idx_nums) + 1)
            idx_nums = np.asarray(idx_nums)
            untied_km[idx_nums] = untied_km[idx_nums]
            num_same = 1
            idx_nums = []
    return untied_km.reshape(self.nobs, 1)