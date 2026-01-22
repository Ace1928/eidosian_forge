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
def _update_mean_params(self):
    """
        Returns
        -------
        update : array_like
            The update vector such that params + update is the next
            iterate when solving the score equations.
        score : array_like
            The current value of the score equations, not
            incorporating the scale parameter.  If desired,
            multiply this vector by the scale parameter to
            incorporate the scale.
        """
    endog = self.endog_li
    exog = self.exog_li
    weights = getattr(self, 'weights_li', None)
    cached_means = self.cached_means
    varfunc = self.family.variance
    bmat, score = (0, 0)
    for i in range(self.num_group):
        expval, lpr = cached_means[i]
        resid = endog[i] - expval
        dmat = self.mean_deriv(exog[i], lpr)
        sdev = np.sqrt(varfunc(expval))
        if weights is not None:
            w = weights[i]
            wresid = resid * w
            wdmat = dmat * w[:, None]
        else:
            wresid = resid
            wdmat = dmat
        rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (wdmat, wresid))
        if rslt is None:
            return (None, None)
        vinv_d, vinv_resid = tuple(rslt)
        bmat += np.dot(dmat.T, vinv_d)
        score += np.dot(dmat.T, vinv_resid)
    try:
        update = np.linalg.solve(bmat, score)
    except np.linalg.LinAlgError:
        update = np.dot(np.linalg.pinv(bmat), score)
    self._fit_history['cov_adjust'].append(self.cov_struct.cov_adjust)
    return (update, score)