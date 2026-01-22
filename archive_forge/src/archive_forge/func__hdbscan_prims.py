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
def _hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric='euclidean', leaf_size=40, n_jobs=None, **metric_params):
    """
    Builds a single-linkage tree (SLT) from the input data `X`. If
    `metric="precomputed"` then `X` must be a symmetric array of distances.
    Otherwise, the pairwise distances are calculated directly and passed to
    `mutual_reachability_graph`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The raw data.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. `metric` must be one of the options allowed by
        :func:`~sklearn.metrics.pairwise_distances` for its metric
        parameter.

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
    X = np.asarray(X, order='C')
    nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm=algo, leaf_size=leaf_size, metric=metric, metric_params=metric_params, n_jobs=n_jobs, p=None).fit(X)
    neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True)
    core_distances = np.ascontiguousarray(neighbors_distances[:, -1])
    dist_metric = DistanceMetric.get_metric(metric, **metric_params)
    min_spanning_tree = mst_from_data_matrix(X, core_distances, dist_metric, alpha)
    return _process_mst(min_spanning_tree)