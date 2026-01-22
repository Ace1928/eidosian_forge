from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def _bc_covmat(self, cov_naive):
    cov_naive = cov_naive / self.scaling_factor
    endog = self.endog_li
    exog = self.exog_li
    varfunc = self.family.variance
    cached_means = self.cached_means
    scale = self.estimate_scale()
    bcm = 0
    for i in range(self.num_group):
        expval, lpr = cached_means[i]
        resid = endog[i] - expval
        dmat = self.mean_deriv(exog[i], lpr)
        sdev = np.sqrt(varfunc(expval))
        rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (dmat,))
        if rslt is None:
            return None
        vinv_d = rslt[0]
        vinv_d /= scale
        hmat = np.dot(vinv_d, cov_naive)
        hmat = np.dot(hmat, dmat.T).T
        f = self.weights_li[i] if self.weights is not None else 1.0
        aresid = np.linalg.solve(np.eye(len(resid)) - hmat, resid)
        rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (aresid,))
        if rslt is None:
            return None
        srt = rslt[0]
        srt = f * np.dot(dmat.T, srt) / scale
        bcm += np.outer(srt, srt)
    cov_robust_bc = np.dot(cov_naive, np.dot(bcm, cov_naive))
    cov_robust_bc *= self.scaling_factor
    return cov_robust_bc