import operator
import sys
import time
import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import _fit_context
from ..exceptions import ConvergenceWarning
from ..linear_model import _cd_fast as cd_fast  # type: ignore
from ..linear_model import lars_path_gram
from ..model_selection import check_cv, cross_val_score
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import EmpiricalCovariance, empirical_covariance, log_likelihood
Fit the GraphicalLasso covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        