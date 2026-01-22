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
def _print_iteration_stats(self, iteration_start_time):
    """Print info about the current fitting iteration."""
    log_msg = ''
    predictors_of_ith_iteration = [predictors_list for predictors_list in self._predictors[-1] if predictors_list]
    n_trees = len(predictors_of_ith_iteration)
    max_depth = max((predictor.get_max_depth() for predictor in predictors_of_ith_iteration))
    n_leaves = sum((predictor.get_n_leaf_nodes() for predictor in predictors_of_ith_iteration))
    if n_trees == 1:
        log_msg += '{} tree, {} leaves, '.format(n_trees, n_leaves)
    else:
        log_msg += '{} trees, {} leaves '.format(n_trees, n_leaves)
        log_msg += '({} on avg), '.format(int(n_leaves / n_trees))
    log_msg += 'max depth = {}, '.format(max_depth)
    if self.do_early_stopping_:
        if self.scoring == 'loss':
            factor = -1
            name = 'loss'
        else:
            factor = 1
            name = 'score'
        log_msg += 'train {}: {:.5f}, '.format(name, factor * self.train_score_[-1])
        if self._use_validation_data:
            log_msg += 'val {}: {:.5f}, '.format(name, factor * self.validation_score_[-1])
    iteration_time = time() - iteration_start_time
    log_msg += 'in {:0.3f}s'.format(iteration_time)
    print(log_msg)