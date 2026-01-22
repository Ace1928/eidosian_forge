from abc import abstractmethod
from typing import List
import numpy as np
from scipy.sparse import issparse
from ... import get_config
from .._dist_metrics import (
from ._argkmin import (
from ._argkmin_classmode import (
from ._base import _sqeuclidean_row_norms32, _sqeuclidean_row_norms64
from ._radius_neighbors import (
from ._radius_neighbors_classmode import (
Return the results of the reduction for the given arguments.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            The input array to be labelled.
        Y : ndarray of shape (n_samples_Y, n_features)
            The input array whose class membership is provided through
            the `Y_labels` parameter.
        radius : float
            The radius defining the neighborhood.
        weights : ndarray
            The weights applied to the `Y_labels` when computing the
            weighted mode of the labels.
        Y_labels : ndarray
            An array containing the index of the class membership of the
            associated samples in `Y`. This is used in labeling `X`.
        unique_Y_labels : ndarray
            An array containing all unique class labels.
        outlier_label : int, default=None
            Label for outlier samples (samples with no neighbors in given
            radius). In the default case when the value is None if any
            outlier is detected, a ValueError will be raised. The outlier
            label should be selected from among the unique 'Y' labels. If
            it is specified with a different value a warning will be raised
            and all class probabilities of outliers will be assigned to be 0.
        metric : str, default='euclidean'
            The distance metric to use. For a list of available metrics, see
            the documentation of :class:`~sklearn.metrics.DistanceMetric`.
            Currently does not support `'precomputed'`.
        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.
        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.
        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.
            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:
              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.
              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.
              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.
              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.
        Returns
        -------
        probabilities : ndarray of shape (n_samples_X, n_classes)
            An array containing the class probabilities for each sample.
        