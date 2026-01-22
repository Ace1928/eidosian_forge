from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def _estimate_vecm_ml(self):
    y_1_T, delta_y_1_T, y_lag1, delta_x = _endog_matrices(self.y, self.exog, self.exog_coint, self.k_ar_diff, self.deterministic, self.seasons, self.first_season)
    T = y_1_T.shape[1]
    s00, s01, s10, s11, s11_, _, v = _sij(delta_x, delta_y_1_T, y_lag1)
    beta_tilde = v[:, :self.coint_rank].T.dot(s11_).T
    beta_tilde = np.real_if_close(beta_tilde)
    beta_tilde = np.dot(beta_tilde, inv(beta_tilde[:self.coint_rank]))
    alpha_tilde = s01.dot(beta_tilde).dot(inv(beta_tilde.T.dot(s11).dot(beta_tilde)))
    gamma_tilde = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_lag1)).dot(delta_x.T).dot(inv(np.dot(delta_x, delta_x.T)))
    temp = delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_lag1) - gamma_tilde.dot(delta_x)
    sigma_u_tilde = temp.dot(temp.T) / T
    return VECMResults(self.y, self.exog, self.exog_coint, self.k_ar, self.coint_rank, alpha_tilde, beta_tilde, gamma_tilde, sigma_u_tilde, deterministic=self.deterministic, seasons=self.seasons, delta_y_1_T=delta_y_1_T, y_lag1=y_lag1, delta_x=delta_x, model=self, names=self.endog_names, dates=self.data.dates, first_season=self.first_season)