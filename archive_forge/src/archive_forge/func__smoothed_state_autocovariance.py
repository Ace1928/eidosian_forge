import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def _smoothed_state_autocovariance(self, shift, start, end, extend_kwargs=None):
    """
        Compute "forward" autocovariances, Cov(t, t+j)

        Parameters
        ----------
        shift : int
            The number of period to shift forwards when computing the
            autocovariance. This has the opposite sign as `lag` from the
            `smoothed_state_autocovariance` method.
        start : int, optional
            The start of the interval (inclusive) of autocovariances to compute
            and return.
        end : int, optional
            The end of the interval (exclusive) autocovariances to compute and
            return. Note that since it is an exclusive endpoint, the returned
            autocovariances do not include the value at this index.
        extend_kwargs : dict, optional
            Keyword arguments containing updated state space system matrices
            for handling out-of-sample autocovariance computations in
            time-varying state space models.

        """
    if extend_kwargs is None:
        extend_kwargs = {}
    n = end - start
    if shift == 0:
        max_insample = self.nobs - shift
    else:
        max_insample = self.nobs - shift + 1
    n_postsample = max(0, end - max_insample)
    if shift != 0:
        L = self.innovations_transition
        P = self.predicted_state_cov
        N = self.scaled_smoothed_estimator_cov
    else:
        acov = self.smoothed_state_cov
    if n_postsample > 0:
        endog = np.zeros((n_postsample, self.k_endog)) * np.nan
        mod = self.model.extend(endog, start=self.nobs, **extend_kwargs)
        mod.initialize_known(self.predicted_state[..., self.nobs], self.predicted_state_cov[..., self.nobs])
        res = mod.smooth()
        if shift != 0:
            start_insample = max(0, start)
            L = np.concatenate((L[..., start_insample:], res.innovations_transition), axis=2)
            P = np.concatenate((P[..., start_insample:], res.predicted_state_cov[..., 1:]), axis=2)
            N = np.concatenate((N[..., start_insample:], res.scaled_smoothed_estimator_cov), axis=2)
            end -= start_insample
            start -= start_insample
        else:
            acov = np.concatenate((acov, res.predicted_state_cov), axis=2)
    if shift != 0:
        start_insample = max(0, start)
        LT = L[..., start_insample:end + shift - 1].T
        P = P[..., start_insample:end + shift].T
        N = N[..., start_insample:end + shift - 1].T
        tmpLT = np.eye(self.k_states)[None, :, :]
        length = P.shape[0] - shift
        for i in range(1, shift + 1):
            tmpLT = LT[shift - i:length + shift - i] @ tmpLT
        eye = np.eye(self.k_states)[None, ...]
        acov = np.zeros((n, self.k_states, self.k_states))
        acov[:start_insample - start] = np.nan
        acov[start_insample - start:] = P[:-shift] @ tmpLT @ (eye - N[shift - 1:] @ P[shift:])
    else:
        acov = acov.T[start:end]
    return acov