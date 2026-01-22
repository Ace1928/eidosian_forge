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
def _update_regularized(self, params, pen_wt, scad_param, eps):
    sn, hm = (0, 0)
    for i in range(self.num_group):
        expval, _ = self.cached_means[i]
        resid = self.endog_li[i] - expval
        sdev = np.sqrt(self.family.variance(expval))
        ex = self.exog_li[i] * sdev[:, None] ** 2
        rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (resid, ex))
        sn0 = rslt[0]
        sn += np.dot(ex.T, sn0)
        hm0 = rslt[1]
        hm += np.dot(ex.T, hm0)
    ap = np.abs(params)
    clipped = np.clip(scad_param * pen_wt - ap, 0, np.inf)
    en = pen_wt * clipped * (ap > pen_wt)
    en /= (scad_param - 1) * pen_wt
    en += pen_wt * (ap <= pen_wt)
    en /= eps + ap
    hm.flat[::hm.shape[0] + 1] += self.num_group * en
    sn -= self.num_group * en * params
    try:
        update = np.linalg.solve(hm, sn)
    except np.linalg.LinAlgError:
        update = np.dot(np.linalg.pinv(hm), sn)
        msg = 'Encountered singularity in regularized GEE update'
        warnings.warn(msg)
    hm *= self.estimate_scale()
    return (update, hm)