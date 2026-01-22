import itertools
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, suppress
from functools import partial
from numbers import Integral, Real
from time import time
import numpy as np
from ..._loss.loss import (
from ...base import (
from ...compose import ColumnTransformer
from ...metrics import check_scoring
from ...metrics._scorer import _SCORERS
from ...model_selection import train_test_split
from ...preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from ...utils import check_random_state, compute_sample_weight, is_scalar_nan, resample
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ...utils.multiclass import check_classification_targets
from ...utils.validation import (
from ._gradient_boosting import _update_raw_predictions
from .binning import _BinMapper
from .common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from .grower import TreeGrower
def _compute_partial_dependence_recursion(self, grid, target_features):
    """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray, shape                 (n_trees_per_iteration, n_samples)
            The value of the partial dependence function on each grid point.
        """
    if getattr(self, '_fitted_with_sw', False):
        raise NotImplementedError("{} does not support partial dependence plots with the 'recursion' method when sample weights were given during fit time.".format(self.__class__.__name__))
    grid = np.asarray(grid, dtype=X_DTYPE, order='C')
    averaged_predictions = np.zeros((self.n_trees_per_iteration_, grid.shape[0]), dtype=Y_DTYPE)
    for predictors_of_ith_iteration in self._predictors:
        for k, predictor in enumerate(predictors_of_ith_iteration):
            predictor.compute_partial_dependence(grid, target_features, averaged_predictions[k])
    return averaged_predictions