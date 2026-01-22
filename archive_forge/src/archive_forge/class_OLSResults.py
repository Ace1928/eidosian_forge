from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
class OLSResults(RegressionResults):
    """
    Results class for for an OLS model.

    Parameters
    ----------
    model : RegressionModel
        The regression model instance.
    params : ndarray
        The estimated parameters.
    normalized_cov_params : ndarray
        The normalized covariance parameters.
    scale : float
        The estimated scale of the residuals.
    cov_type : str
        The covariance estimator used in the results.
    cov_kwds : dict
        Additional keywords used in the covariance specification.
    use_t : bool
        Flag indicating to use the Student's t in inference.
    **kwargs
        Additional keyword arguments used to initialize the results.

    See Also
    --------
    RegressionResults
        Results store for WLS and GLW models.

    Notes
    -----
    Most of the methods and attributes are inherited from RegressionResults.
    The special methods that are only available for OLS are:

    - get_influence
    - outlier_test
    - el_test
    - conf_int_el
    """

    def get_influence(self):
        """
        Calculate influence and outlier measures.

        Returns
        -------
        OLSInfluence
            The instance containing methods to calculate the main influence and
            outlier measures for the OLS regression.

        See Also
        --------
        statsmodels.stats.outliers_influence.OLSInfluence
            A class that exposes methods to examine observation influence.
        """
        from statsmodels.stats.outliers_influence import OLSInfluence
        return OLSInfluence(self)

    def outlier_test(self, method='bonf', alpha=0.05, labels=None, order=False, cutoff=None):
        """
        Test observations for outliers according to method.

        Parameters
        ----------
        method : str
            The method to use in the outlier test.  Must be one of:

            - `bonferroni` : one-step correction
            - `sidak` : one-step correction
            - `holm-sidak` :
            - `holm` :
            - `simes-hochberg` :
            - `hommel` :
            - `fdr_bh` : Benjamini/Hochberg
            - `fdr_by` : Benjamini/Yekutieli

            See `statsmodels.stats.multitest.multipletests` for details.
        alpha : float
            The familywise error rate (FWER).
        labels : None or array_like
            If `labels` is not None, then it will be used as index to the
            returned pandas DataFrame. See also Returns below.
        order : bool
            Whether or not to order the results by the absolute value of the
            studentized residuals. If labels are provided they will also be
            sorted.
        cutoff : None or float in [0, 1]
            If cutoff is not None, then the return only includes observations
            with multiple testing corrected p-values strictly below the cutoff.
            The returned array or dataframe can be empty if t.

        Returns
        -------
        array_like
            Returns either an ndarray or a DataFrame if labels is not None.
            Will attempt to get labels from model_results if available. The
            columns are the Studentized residuals, the unadjusted p-value,
            and the corrected p-value according to method.

        Notes
        -----
        The unadjusted p-value is stats.t.sf(abs(resid), df) where
        df = df_resid - 1.
        """
        from statsmodels.stats.outliers_influence import outlier_test
        return outlier_test(self, method, alpha, labels=labels, order=order, cutoff=cutoff)

    def el_test(self, b0_vals, param_nums, return_weights=0, ret_params=0, method='nm', stochastic_exog=1):
        """
        Test single or joint hypotheses using Empirical Likelihood.

        Parameters
        ----------
        b0_vals : 1darray
            The hypothesized value of the parameter to be tested.
        param_nums : 1darray
            The parameter number to be tested.
        return_weights : bool
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals. The default is False.
        ret_params : bool
            If true, returns the parameter vector that maximizes the likelihood
            ratio at b0_vals.  Also returns the weights.  The default is False.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors. The default is True.

        Returns
        -------
        tuple
            The p-value and -2 times the log-likelihood ratio for the
            hypothesized values.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.stackloss.load()
        >>> endog = data.endog
        >>> exog = sm.add_constant(data.exog)
        >>> model = sm.OLS(endog, exog)
        >>> fitted = model.fit()
        >>> fitted.params
        >>> array([-39.91967442,   0.7156402 ,   1.29528612,  -0.15212252])
        >>> fitted.rsquared
        >>> 0.91357690446068196
        >>> # Test that the slope on the first variable is 0
        >>> fitted.el_test([0], [1])
        >>> (27.248146353888796, 1.7894660442330235e-07)
        """
        params = np.copy(self.params)
        opt_fun_inst = _ELRegOpts()
        if len(param_nums) == len(params):
            llr = opt_fun_inst._opt_nuis_regress([], param_nums=param_nums, endog=self.model.endog, exog=self.model.exog, nobs=self.model.nobs, nvar=self.model.exog.shape[1], params=params, b0_vals=b0_vals, stochastic_exog=stochastic_exog)
            pval = 1 - stats.chi2.cdf(llr, len(param_nums))
            if return_weights:
                return (llr, pval, opt_fun_inst.new_weights)
            else:
                return (llr, pval)
        x0 = np.delete(params, param_nums)
        args = (param_nums, self.model.endog, self.model.exog, self.model.nobs, self.model.exog.shape[1], params, b0_vals, stochastic_exog)
        if method == 'nm':
            llr = optimize.fmin(opt_fun_inst._opt_nuis_regress, x0, maxfun=10000, maxiter=10000, full_output=1, disp=0, args=args)[1]
        if method == 'powell':
            llr = optimize.fmin_powell(opt_fun_inst._opt_nuis_regress, x0, full_output=1, disp=0, args=args)[1]
        pval = 1 - stats.chi2.cdf(llr, len(param_nums))
        if ret_params:
            return (llr, pval, opt_fun_inst.new_weights, opt_fun_inst.new_params)
        elif return_weights:
            return (llr, pval, opt_fun_inst.new_weights)
        else:
            return (llr, pval)

    def conf_int_el(self, param_num, sig=0.05, upper_bound=None, lower_bound=None, method='nm', stochastic_exog=True):
        """
        Compute the confidence interval using Empirical Likelihood.

        Parameters
        ----------
        param_num : float
            The parameter for which the confidence interval is desired.
        sig : float
            The significance level.  Default is 0.05.
        upper_bound : float
            The maximum value the upper limit can be.  Default is the
            99.9% confidence value under OLS assumptions.
        lower_bound : float
            The minimum value the lower limit can be.  Default is the 99.9%
            confidence value under OLS assumptions.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  The default is True.

        Returns
        -------
        lowerl : float
            The lower bound of the confidence interval.
        upperl : float
            The upper bound of the confidence interval.

        See Also
        --------
        el_test : Test parameters using Empirical Likelihood.

        Notes
        -----
        This function uses brentq to find the value of beta where
        test_beta([beta], param_num)[1] is equal to the critical value.

        The function returns the results of each iteration of brentq at each
        value of beta.

        The current function value of the last printed optimization should be
        the critical value at the desired significance level. For alpha=.05,
        the value is 3.841459.

        To ensure optimization terminated successfully, it is suggested to do
        el_test([lower_limit], [param_num]).

        If the optimization does not terminate successfully, consider switching
        optimization algorithms.

        If optimization is still not successful, try changing the values of
        start_int_params.  If the current function value repeatedly jumps
        from a number between 0 and the critical value and a very large number
        (>50), the starting parameters of the interior minimization need
        to be changed.
        """
        r0 = stats.chi2.ppf(1 - sig, 1)
        if upper_bound is None:
            upper_bound = self.conf_int(0.01)[param_num][1]
        if lower_bound is None:
            lower_bound = self.conf_int(0.01)[param_num][0]

        def f(b0):
            return self.el_test(np.array([b0]), np.array([param_num]), method=method, stochastic_exog=stochastic_exog)[0] - r0
        lowerl = optimize.brenth(f, lower_bound, self.params[param_num])
        upperl = optimize.brenth(f, self.params[param_num], upper_bound)
        return (lowerl, upperl)