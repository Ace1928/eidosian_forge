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
def _update_leaves_values(loss, grower, y_true, raw_prediction, sample_weight):
    """Update the leaf values to be predicted by the tree.

    Update equals:
        loss.fit_intercept_only(y_true - raw_prediction)

    This is only applied if loss.differentiable is False.
    Note: It only works, if the loss is a function of the residual, as is the
    case for AbsoluteError and PinballLoss. Otherwise, one would need to get
    the minimum of loss(y_true, raw_prediction + x) in x. A few examples:
      - AbsoluteError: median(y_true - raw_prediction).
      - PinballLoss: quantile(y_true - raw_prediction).

    More background:
    For the standard gradient descent method according to "Greedy Function
    Approximation: A Gradient Boosting Machine" by Friedman, all loss functions but the
    squared loss need a line search step. BaseHistGradientBoosting, however, implements
    a so called Newton boosting where the trees are fitted to a 2nd order
    approximations of the loss in terms of gradients and hessians. In this case, the
    line search step is only necessary if the loss is not smooth, i.e. not
    differentiable, which renders the 2nd order approximation invalid. In fact,
    non-smooth losses arbitrarily set hessians to 1 and effectively use the standard
    gradient descent method with line search.
    """
    for leaf in grower.finalized_leaves:
        indices = leaf.sample_indices
        if sample_weight is None:
            sw = None
        else:
            sw = sample_weight[indices]
        update = loss.fit_intercept_only(y_true=y_true[indices] - raw_prediction[indices], sample_weight=sw)
        leaf.value = grower.shrinkage * update