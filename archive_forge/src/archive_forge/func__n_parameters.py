import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _n_parameters(self):
    """Return the number of free parameters in the model."""
    _, n_features = self.means_.shape
    if self.covariance_type == 'full':
        cov_params = self.n_components * n_features * (n_features + 1) / 2.0
    elif self.covariance_type == 'diag':
        cov_params = self.n_components * n_features
    elif self.covariance_type == 'tied':
        cov_params = n_features * (n_features + 1) / 2.0
    elif self.covariance_type == 'spherical':
        cov_params = self.n_components
    mean_params = n_features * self.n_components
    return int(cov_params + mean_params + self.n_components - 1)