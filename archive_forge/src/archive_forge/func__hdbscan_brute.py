from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import csgraph, issparse
from ...base import BaseEstimator, ClusterMixin, _fit_context
from ...metrics import pairwise_distances
from ...metrics._dist_metrics import DistanceMetric
from ...neighbors import BallTree, KDTree, NearestNeighbors
from ...utils._param_validation import Interval, StrOptions
from ...utils.validation import _allclose_dense_sparse, _assert_all_finite
from ._linkage import (
from ._reachability import mutual_reachability_graph
from ._tree import HIERARCHY_dtype, labelling_at_cut, tree_to_labels
def _hdbscan_brute(X, min_samples=5, alpha=None, metric='euclidean', n_jobs=None, copy=False, **metric_params):
    """
    Builds a single-linkage tree (SLT) from the input data `X`. If
    `metric="precomputed"` then `X` must be a symmetric array of distances.
    Otherwise, the pairwise distances are calculated directly and passed to
    `mutual_reachability_graph`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        Either the raw data from which to compute the pairwise distances,
        or the precomputed distances.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.

        - If metric is a string or callable, it must be one of
          the options allowed by :func:`~sklearn.metrics.pairwise_distances`
          for its metric parameter.

        - If metric is "precomputed", X is assumed to be a distance matrix and
          must be square.

    n_jobs : int, default=None
        The number of jobs to use for computing the pairwise distances. This
        works by breaking down the pairwise matrix into n_jobs even slices and
        computing them in parallel. This parameter is passed directly to
        :func:`~sklearn.metrics.pairwise_distances`.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    copy : bool, default=False
        If `copy=True` then any time an in-place modifications would be made
        that would overwrite `X`, a copy will first be made, guaranteeing that
        the original data will be unchanged. Currently, it only applies when
        `metric="precomputed"`, when passing a dense array or a CSR sparse
        array/matrix.

    metric_params : dict, default=None
        Arguments passed to the distance metric.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    if metric == 'precomputed':
        if X.shape[0] != X.shape[1]:
            raise ValueError(f'The precomputed distance matrix is expected to be symmetric, however it has shape {X.shape}. Please verify that the distance matrix was constructed correctly.')
        if not _allclose_dense_sparse(X, X.T):
            raise ValueError('The precomputed distance matrix is expected to be symmetric, however its values appear to be asymmetric. Please verify that the distance matrix was constructed correctly.')
        distance_matrix = X.copy() if copy else X
    else:
        distance_matrix = pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)
    distance_matrix /= alpha
    max_distance = metric_params.get('max_distance', 0.0)
    if issparse(distance_matrix) and distance_matrix.format != 'csr':
        distance_matrix = distance_matrix.tocsr()
    mutual_reachability_ = mutual_reachability_graph(distance_matrix, min_samples=min_samples, max_distance=max_distance)
    min_spanning_tree = _brute_mst(mutual_reachability_, min_samples=min_samples)
    if np.isinf(min_spanning_tree['distance']).any():
        warn('The minimum spanning tree contains edge weights with value infinity. Potentially, you are missing too many distances in the initial distance matrix for the given neighborhood size.', UserWarning)
    return _process_mst(min_spanning_tree)