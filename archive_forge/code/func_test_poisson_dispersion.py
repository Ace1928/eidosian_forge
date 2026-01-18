import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def test_poisson_dispersion(results, method='all', _old=False):
    """Score/LM type tests for Poisson variance assumptions

    Null Hypothesis is

    H0: var(y) = E(y) and assuming E(y) is correctly specified
    H1: var(y) ~= E(y)

    The tests are based on the constrained model, i.e. the Poisson model.
    The tests differ in their assumed alternatives, and in their maintained
    assumptions.

    Parameters
    ----------
    results : Poisson results instance
        This can be a results instance for either a discrete Poisson or a GLM
        with family Poisson.
    method : str
        Not used yet. Currently results for all methods are returned.
    _old : bool
        Temporary keyword for backwards compatibility, will be removed
        in future version of statsmodels.

    Returns
    -------
    res : instance
        The instance of DispersionResults has the hypothesis test results,
        statistic, pvalue, method, alternative, as main attributes and a
        summary_frame method that returns the results as pandas DataFrame.

    """
    if method not in ['all']:
        raise ValueError(f'unknown method "{method}"')
    if hasattr(results, '_results'):
        results = results._results
    endog = results.model.endog
    nobs = endog.shape[0]
    fitted = results.predict()
    resid2 = results.resid_response ** 2
    var_resid_endog = resid2 - endog
    var_resid_fitted = resid2 - fitted
    std1 = np.sqrt(2 * (fitted ** 2).sum())
    var_resid_endog_sum = var_resid_endog.sum()
    dean_a = var_resid_fitted.sum() / std1
    dean_b = var_resid_endog_sum / std1
    dean_c = (var_resid_endog / fitted).sum() / np.sqrt(2 * nobs)
    pval_dean_a = 2 * stats.norm.sf(np.abs(dean_a))
    pval_dean_b = 2 * stats.norm.sf(np.abs(dean_b))
    pval_dean_c = 2 * stats.norm.sf(np.abs(dean_c))
    results_all = [[dean_a, pval_dean_a], [dean_b, pval_dean_b], [dean_c, pval_dean_c]]
    description = [['Dean A', 'mu (1 + a mu)'], ['Dean B', 'mu (1 + a mu)'], ['Dean C', 'mu (1 + a)']]
    endog_v = var_resid_endog / fitted
    res_ols_nb2 = OLS(endog_v, fitted).fit(use_t=False)
    stat_ols_nb2 = res_ols_nb2.tvalues[0]
    pval_ols_nb2 = res_ols_nb2.pvalues[0]
    results_all.append([stat_ols_nb2, pval_ols_nb2])
    description.append(['CT nb2', 'mu (1 + a mu)'])
    res_ols_nb1 = OLS(endog_v, fitted).fit(use_t=False)
    stat_ols_nb1 = res_ols_nb1.tvalues[0]
    pval_ols_nb1 = res_ols_nb1.pvalues[0]
    results_all.append([stat_ols_nb1, pval_ols_nb1])
    description.append(['CT nb1', 'mu (1 + a)'])
    endog_v = var_resid_endog / fitted
    res_ols_nb2 = OLS(endog_v, fitted).fit(cov_type='HC3', use_t=False)
    stat_ols_hc1_nb2 = res_ols_nb2.tvalues[0]
    pval_ols_hc1_nb2 = res_ols_nb2.pvalues[0]
    results_all.append([stat_ols_hc1_nb2, pval_ols_hc1_nb2])
    description.append(['CT nb2 HC3', 'mu (1 + a mu)'])
    res_ols_nb1 = OLS(endog_v, np.ones(len(endog_v))).fit(cov_type='HC3', use_t=False)
    stat_ols_hc1_nb1 = res_ols_nb1.tvalues[0]
    pval_ols_hc1_nb1 = res_ols_nb1.pvalues[0]
    results_all.append([stat_ols_hc1_nb1, pval_ols_hc1_nb1])
    description.append(['CT nb1 HC3', 'mu (1 + a)'])
    results_all = np.array(results_all)
    if _old:
        return (results_all, description)
    else:
        res = DispersionResults(statistic=results_all[:, 0], pvalue=results_all[:, 1], method=[i[0] for i in description], alternative=[i[1] for i in description], name='Poisson Dispersion Test')
        return res