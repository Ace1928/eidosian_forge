import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def _fit_em(self, start_params=None, transformed=True, cov_type='none', cov_kwds=None, maxiter=50, tolerance=1e-06, full_output=True, return_params=False, **kwargs):
    """
        Fits the model using the Expectation-Maximization (EM) algorithm

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by `start_params`.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The type of covariance matrix estimator to use. Can be one of
            'approx', 'opg', 'robust', or 'none'. Default is 'none'.
        cov_kwds : dict or None, optional
            Keywords for alternative covariance estimators
        maxiter : int, optional
            The maximum number of iterations to perform.
        tolerance : float, optional
            The iteration stops when the difference between subsequent
            loglikelihood values is less than this tolerance.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. This includes all intermediate values for
            parameters and loglikelihood values
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring. It has not been tested for a thoroughly correct EM
        implementation in all cases. It does not support TVTP transition
        probabilities.

        Returns
        -------
        MarkovSwitchingResults
        """
    if start_params is None:
        start_params = self.start_params
        transformed = True
    else:
        start_params = np.array(start_params, ndmin=1)
    if not transformed:
        start_params = self.transform_params(start_params)
    llf = []
    params = [start_params]
    i = 0
    delta = 0
    while i < maxiter and (i < 2 or delta > tolerance):
        out = self._em_iteration(params[-1])
        llf.append(out[0].llf)
        params.append(out[1])
        if i > 0:
            delta = 2 * (llf[-1] - llf[-2]) / np.abs(llf[-1] + llf[-2])
        i += 1
    if return_params:
        result = params[-1]
    else:
        result = self.filter(params[-1], transformed=True, cov_type=cov_type, cov_kwds=cov_kwds)
        if full_output:
            em_retvals = Bunch(**{'params': np.array(params), 'llf': np.array(llf), 'iter': i})
            em_settings = Bunch(**{'tolerance': tolerance, 'maxiter': maxiter})
        else:
            em_retvals = None
            em_settings = None
        result.mle_retvals = em_retvals
        result.mle_settings = em_settings
    return result