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
class GroupKFold(GroupsConsumerMixin, _BaseKFold):
    """K-fold iterator variant with non-overlapping groups.

    Each group will appear exactly once in the test set across all folds (the
    number of distinct groups has to be at least equal to the number of folds).

    The folds are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.

    Read more in the :ref:`User Guide <group_k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    Notes
    -----
    Groups appear in an arbitrary order throughout the folds.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([0, 0, 2, 2, 3, 3])
    >>> group_kfold = GroupKFold(n_splits=2)
    >>> group_kfold.get_n_splits(X, y, groups)
    2
    >>> print(group_kfold)
    GroupKFold(n_splits=2)
    >>> for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3], group=[2 2]
      Test:  index=[0 1 4 5], group=[0 0 3 3]
    Fold 1:
      Train: index=[0 1 4 5], group=[0 0 3 3]
      Test:  index=[2 3], group=[2 2]

    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit
        domain-specific stratification of the dataset.

    StratifiedKFold : Takes class information into account to avoid building
        folds with imbalanced class proportions (for binary or multiclass
        classification tasks).
    """

    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name='groups', ensure_2d=False, dtype=None)
        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        if self.n_splits > n_groups:
            raise ValueError('Cannot have number of splits n_splits=%d greater than the number of groups: %d.' % (self.n_splits, n_groups))
        n_samples_per_group = np.bincount(groups)
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]
        n_samples_per_fold = np.zeros(self.n_splits)
        group_to_fold = np.zeros(len(unique_groups))
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold
        indices = group_to_fold[groups]
        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)