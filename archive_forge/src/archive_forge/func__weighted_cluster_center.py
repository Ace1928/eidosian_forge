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
def _weighted_cluster_center(self, X):
    """Calculate and store the centroids/medoids of each cluster.

        This requires `X` to be a raw feature array, not precomputed
        distances. Rather than return outputs directly, this helper method
        instead stores them in the `self.{centroids, medoids}_` attributes.
        The choice for which attributes are calculated and stored is mediated
        by the value of `self.store_centers`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature array that the estimator was fit with.

        """
    n_clusters = len(set(self.labels_) - {-1, -2})
    mask = np.empty((X.shape[0],), dtype=np.bool_)
    make_centroids = self.store_centers in ('centroid', 'both')
    make_medoids = self.store_centers in ('medoid', 'both')
    if make_centroids:
        self.centroids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
    if make_medoids:
        self.medoids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
    for idx in range(n_clusters):
        mask = self.labels_ == idx
        data = X[mask]
        strength = self.probabilities_[mask]
        if make_centroids:
            self.centroids_[idx] = np.average(data, weights=strength, axis=0)
        if make_medoids:
            dist_mat = pairwise_distances(data, metric=self.metric, **self._metric_params)
            dist_mat = dist_mat * strength
            medoid_index = np.argmin(dist_mat.sum(axis=1))
            self.medoids_[idx] = data[medoid_index]
    return