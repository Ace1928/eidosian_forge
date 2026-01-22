import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from inspect import signature
from itertools import chain, combinations
from math import ceil, floor
import numpy as np
from scipy.special import comb
from ..utils import (
from ..utils._param_validation import Interval, RealNotInt, validate_params
from ..utils.metadata_routing import _MetadataRequester
from ..utils.multiclass import type_of_target
from ..utils.validation import _num_samples, check_array, column_or_1d
def _make_test_folds(self, X, y=None):
    rng = check_random_state(self.random_state)
    y = np.asarray(y)
    type_of_target_y = type_of_target(y)
    allowed_target_types = ('binary', 'multiclass')
    if type_of_target_y not in allowed_target_types:
        raise ValueError('Supported target types are: {}. Got {!r} instead.'.format(allowed_target_types, type_of_target_y))
    y = column_or_1d(y)
    _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
    _, class_perm = np.unique(y_idx, return_inverse=True)
    y_encoded = class_perm[y_inv]
    n_classes = len(y_idx)
    y_counts = np.bincount(y_encoded)
    min_groups = np.min(y_counts)
    if np.all(self.n_splits > y_counts):
        raise ValueError('n_splits=%d cannot be greater than the number of members in each class.' % self.n_splits)
    if self.n_splits > min_groups:
        warnings.warn('The least populated class in y has only %d members, which is less than n_splits=%d.' % (min_groups, self.n_splits), UserWarning)
    y_order = np.sort(y_encoded)
    allocation = np.asarray([np.bincount(y_order[i::self.n_splits], minlength=n_classes) for i in range(self.n_splits)])
    test_folds = np.empty(len(y), dtype='i')
    for k in range(n_classes):
        folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
        if self.shuffle:
            rng.shuffle(folds_for_class)
        test_folds[y_encoded == k] = folds_for_class
    return test_folds