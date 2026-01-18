import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def update_smoother(self, smoother):
    """
        Update the smoother results

        Parameters
        ----------
        smoother : KalmanSmoother
            The model object from which to take the updated values.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
    attributes = []
    if self.smoother_state or self.smoother_disturbance:
        attributes.append('scaled_smoothed_estimator')
    if self.smoother_state_cov or self.smoother_disturbance_cov:
        attributes.append('scaled_smoothed_estimator_cov')
    if self.smoother_state:
        attributes.append('smoothed_state')
    if self.smoother_state_cov:
        attributes.append('smoothed_state_cov')
    if self.smoother_state_autocov:
        attributes.append('smoothed_state_autocov')
    if self.smoother_disturbance:
        attributes += ['smoothing_error', 'smoothed_measurement_disturbance', 'smoothed_state_disturbance']
    if self.smoother_disturbance_cov:
        attributes += ['smoothed_measurement_disturbance_cov', 'smoothed_state_disturbance_cov']
    has_missing = np.sum(self.nmissing) > 0
    for name in self._smoother_attributes:
        if name == 'smoother_output':
            pass
        elif name in attributes:
            if name in ['smoothing_error', 'smoothed_measurement_disturbance']:
                vector = getattr(smoother, name, None)
                if vector is not None and has_missing:
                    vector = np.array(reorder_missing_vector(vector, self.missing, prefix=self.prefix))
                else:
                    vector = np.array(vector, copy=True)
                setattr(self, name, vector)
            elif name == 'smoothed_measurement_disturbance_cov':
                matrix = getattr(smoother, name, None)
                if matrix is not None and has_missing:
                    matrix = reorder_missing_matrix(matrix, self.missing, reorder_rows=True, reorder_cols=True, prefix=self.prefix)
                    copy_index_matrix(self.obs_cov, matrix, self.missing, index_rows=True, index_cols=True, inplace=True, prefix=self.prefix)
                else:
                    matrix = np.array(matrix, copy=True)
                setattr(self, name, matrix)
            else:
                setattr(self, name, np.array(getattr(smoother, name, None), copy=True))
        else:
            setattr(self, name, None)
    self.innovations_transition = np.array(smoother.innovations_transition, copy=True)
    self.scaled_smoothed_diffuse_estimator = None
    self.scaled_smoothed_diffuse1_estimator_cov = None
    self.scaled_smoothed_diffuse2_estimator_cov = None
    if self.nobs_diffuse > 0:
        self.scaled_smoothed_diffuse_estimator = np.array(smoother.scaled_smoothed_diffuse_estimator, copy=True)
        self.scaled_smoothed_diffuse1_estimator_cov = np.array(smoother.scaled_smoothed_diffuse1_estimator_cov, copy=True)
        self.scaled_smoothed_diffuse2_estimator_cov = np.array(smoother.scaled_smoothed_diffuse2_estimator_cov, copy=True)
    start = 1
    end = None
    if 'scaled_smoothed_estimator' in attributes:
        self.scaled_smoothed_estimator_presample = self.scaled_smoothed_estimator[:, 0]
        self.scaled_smoothed_estimator = self.scaled_smoothed_estimator[:, start:end]
    if 'scaled_smoothed_estimator_cov' in attributes:
        self.scaled_smoothed_estimator_cov_presample = self.scaled_smoothed_estimator_cov[:, :, 0]
        self.scaled_smoothed_estimator_cov = self.scaled_smoothed_estimator_cov[:, :, start:end]
    self._smoothed_forecasts = None
    self._smoothed_forecasts_error = None
    self._smoothed_forecasts_error_cov = None
    if self.filter_concentrated and self.model._scale is None:
        self.smoothed_state_cov *= self.scale
        self.smoothed_state_autocov *= self.scale
        self.smoothed_state_disturbance_cov *= self.scale
        self.smoothed_measurement_disturbance_cov *= self.scale
        self.scaled_smoothed_estimator_presample /= self.scale
        self.scaled_smoothed_estimator /= self.scale
        self.scaled_smoothed_estimator_cov_presample /= self.scale
        self.scaled_smoothed_estimator_cov /= self.scale
        self.smoothing_error /= self.scale
    self.__smoothed_state_autocovariance = {}