import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def _test_poisson_dispersion_generic(results, exog_new_test, exog_new_control=None, include_score=False, use_endog=True, cov_type='HC3', cov_kwds=None, use_t=False):
    """A variable addition test for the variance function

    This uses an artificial regression to calculate a variant of an LM or
    generalized score test for the specification of the variance assumption
    in a Poisson model. The performed test is a Wald test on the coefficients
    of the `exog_new_test`.

    Warning: insufficiently tested, especially for options
    """
    if hasattr(results, '_results'):
        results = results._results
    endog = results.model.endog
    nobs = endog.shape[0]
    fitted = results.predict()
    resid2 = results.resid_response ** 2
    if use_endog:
        var_resid = resid2 - endog
    else:
        var_resid = resid2 - fitted
    endog_v = var_resid / fitted
    k_constraints = exog_new_test.shape[1]
    ex_list = [exog_new_test]
    if include_score:
        score_obs = results.model.score_obs(results.params)
        ex_list.append(score_obs)
    if exog_new_control is not None:
        ex_list.append(score_obs)
    if len(ex_list) > 1:
        ex = np.column_stack(ex_list)
        use_wald = True
    else:
        ex = ex_list[0]
        use_wald = False
    res_ols = OLS(endog_v, ex).fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    if use_wald:
        k_vars = ex.shape[1]
        constraints = np.eye(k_constraints, k_vars)
        ht = res_ols.wald_test(constraints)
        stat_ols = ht.statistic
        pval_ols = ht.pvalue
    else:
        nobs = endog_v.shape[0]
        rsquared_noncentered = 1 - res_ols.ssr / res_ols.uncentered_tss
        stat_ols = nobs * rsquared_noncentered
        pval_ols = stats.chi2.sf(stat_ols, k_constraints)
    return (stat_ols, pval_ols)