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
def _iter_indices(self, X, y, groups=None):
    n_samples = _num_samples(X)
    y = check_array(y, input_name='y', ensure_2d=False, dtype=None)
    n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size, default_test_size=self._default_test_size)
    if y.ndim == 2:
        y = np.array([' '.join(row.astype('str')) for row in y])
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = np.bincount(y_indices)
    if np.min(class_counts) < 2:
        raise ValueError('The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.')
    if n_train < n_classes:
        raise ValueError('The train_size = %d should be greater or equal to the number of classes = %d' % (n_train, n_classes))
    if n_test < n_classes:
        raise ValueError('The test_size = %d should be greater or equal to the number of classes = %d' % (n_test, n_classes))
    class_indices = np.split(np.argsort(y_indices, kind='mergesort'), np.cumsum(class_counts)[:-1])
    rng = check_random_state(self.random_state)
    for _ in range(self.n_splits):
        n_i = _approximate_mode(class_counts, n_train, rng)
        class_counts_remaining = class_counts - n_i
        t_i = _approximate_mode(class_counts_remaining, n_test, rng)
        train = []
        test = []
        for i in range(n_classes):
            permutation = rng.permutation(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode='clip')
            train.extend(perm_indices_class_i[:n_i[i]])
            test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])
        train = rng.permutation(train)
        test = rng.permutation(test)
        yield (train, test)