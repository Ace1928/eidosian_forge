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
class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <time_series_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    .. versionadded:: 0.18

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value
        with ``gap=0``.

        .. versionadded:: 0.24

    gap : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.

        .. versionadded:: 0.24

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit()
    >>> print(tscv)
    TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0]
      Test:  index=[1]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[2]
    Fold 2:
      Train: index=[0 1 2]
      Test:  index=[3]
    Fold 3:
      Train: index=[0 1 2 3]
      Test:  index=[4]
    Fold 4:
      Train: index=[0 1 2 3 4]
      Test:  index=[5]
    >>> # Fix test_size to 2 with 12 samples
    >>> X = np.random.randn(12, 2)
    >>> y = np.random.randint(0, 2, 12)
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2 3 4 5]
      Test:  index=[6 7]
    Fold 1:
      Train: index=[0 1 2 3 4 5 6 7]
      Test:  index=[8 9]
    Fold 2:
      Train: index=[0 1 2 3 4 5 6 7 8 9]
      Test:  index=[10 11]
    >>> # Add in a 2 period gap
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2 3]
      Test:  index=[6 7]
    Fold 1:
      Train: index=[0 1 2 3 4 5]
      Test:  index=[8 9]
    Fold 2:
      Train: index=[0 1 2 3 4 5 6 7]
      Test:  index=[10 11]

    For a more extended example see
    :ref:`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`.

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i`` th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples.
    """

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = self.test_size if self.test_size is not None else n_samples // n_folds
        if n_folds > n_samples:
            raise ValueError(f'Cannot have number of folds={n_folds} greater than the number of samples={n_samples}.')
        if n_samples - gap - test_size * n_splits <= 0:
            raise ValueError(f'Too many splits={n_splits} for number of samples={n_samples} with test_size={test_size} and gap={gap}.')
        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)
        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                yield (indices[train_end - self.max_train_size:train_end], indices[test_start:test_start + test_size])
            else:
                yield (indices[:train_end], indices[test_start:test_start + test_size])