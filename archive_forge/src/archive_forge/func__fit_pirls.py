from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
def _fit_pirls(self, alpha, start_params=None, maxiter=100, tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, weights=None):
    """fit model with penalized reweighted least squares
        """
    endog = self.endog
    wlsexog = self.exog
    spl_s = self.penal.penalty_matrix(alpha=alpha)
    nobs, n_columns = wlsexog.shape
    if weights is None:
        self.data_weights = np.array([1.0] * nobs)
    else:
        self.data_weights = weights
    if not hasattr(self, '_offset_exposure'):
        self._offset_exposure = 0
    self.scaletype = scale
    self.scale = 1
    if start_params is None:
        mu = self.family.starting_mu(endog)
        lin_pred = self.family.predict(mu)
    else:
        lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
        mu = self.family.fitted(lin_pred)
    dev = self.family.deviance(endog, mu)
    history = dict(params=[None, start_params], deviance=[np.inf, dev])
    converged = False
    criterion = history['deviance']
    if maxiter == 0:
        mu = self.family.fitted(lin_pred)
        self.scale = self.estimate_scale(mu)
        wls_results = lm.RegressionResults(self, start_params, None)
        iteration = 0
    for iteration in range(maxiter):
        self.weights = self.data_weights * self.family.weights(mu)
        wlsendog = lin_pred + self.family.link.deriv(mu) * (endog - mu) - self._offset_exposure
        wls_results = penalized_wls(wlsendog, wlsexog, spl_s, self.weights)
        lin_pred = np.dot(wlsexog, wls_results.params).ravel()
        lin_pred += self._offset_exposure
        mu = self.family.fitted(lin_pred)
        history = self._update_history(wls_results, mu, history)
        if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
            msg = 'Perfect separation detected, results not available'
            raise PerfectSeparationError(msg)
        converged = _check_convergence(criterion, iteration, tol, 0)
        if converged:
            break
    self.mu = mu
    self.scale = self.estimate_scale(mu)
    glm_results = GLMGamResults(self, wls_results.params, wls_results.normalized_cov_params, self.scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    glm_results.method = 'PIRLS'
    history['iteration'] = iteration + 1
    glm_results.fit_history = history
    glm_results.converged = converged
    return GLMGamResultsWrapper(glm_results)