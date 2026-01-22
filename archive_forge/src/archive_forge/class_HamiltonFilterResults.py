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
class HamiltonFilterResults:
    """
    Results from applying the Hamilton filter to a state space model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_regimes : int
        The number of unobserved regimes.
    regime_transition : ndarray
        The regime transition matrix.
    initialization : str
        Initialization method for regime probabilities.
    initial_probabilities : ndarray
        Initial regime probabilities
    conditional_loglikelihoods : ndarray
        The loglikelihood values at each time period, conditional on regime.
    predicted_joint_probabilities : ndarray
        Predicted joint probabilities at each time period.
    filtered_marginal_probabilities : ndarray
        Filtered marginal probabilities at each time period.
    filtered_joint_probabilities : ndarray
        Filtered joint probabilities at each time period.
    joint_loglikelihoods : ndarray
        The likelihood values at each time period.
    llf_obs : ndarray
        The loglikelihood values at each time period.
    """

    def __init__(self, model, result):
        self.model = model
        self.nobs = model.nobs
        self.order = model.order
        self.k_regimes = model.k_regimes
        attributes = ['regime_transition', 'initial_probabilities', 'conditional_loglikelihoods', 'predicted_joint_probabilities', 'filtered_marginal_probabilities', 'filtered_joint_probabilities', 'joint_loglikelihoods']
        for name in attributes:
            setattr(self, name, getattr(result, name))
        self.initialization = model._initialization
        self.llf_obs = self.joint_loglikelihoods
        self.llf = np.sum(self.llf_obs)
        if self.regime_transition.shape[-1] > 1 and self.order > 0:
            self.regime_transition = self.regime_transition[..., self.order:]
        self._predicted_marginal_probabilities = None

    @property
    def predicted_marginal_probabilities(self):
        if self._predicted_marginal_probabilities is None:
            self._predicted_marginal_probabilities = self.predicted_joint_probabilities
            for i in range(self._predicted_marginal_probabilities.ndim - 2):
                self._predicted_marginal_probabilities = np.sum(self._predicted_marginal_probabilities, axis=-2)
        return self._predicted_marginal_probabilities

    @property
    def expected_durations(self):
        """
        (array) Expected duration of a regime, possibly time-varying.
        """
        diag = np.diagonal(self.regime_transition)
        expected_durations = np.zeros_like(diag)
        degenerate = np.any(diag == 1, axis=1)
        expected_durations[~degenerate] = 1 / (1 - diag[~degenerate])
        expected_durations[degenerate] = np.nan
        expected_durations[diag == 1] = np.inf
        return expected_durations.squeeze()