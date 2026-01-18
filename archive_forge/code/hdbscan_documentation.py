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
Return clustering given by DBSCAN without border points.

        Return clustering that would be equivalent to running DBSCAN* for a
        particular cut_distance (or epsilon) DBSCAN* can be thought of as
        DBSCAN without the border points.  As such these results may differ
        slightly from `cluster.DBSCAN` due to the difference in implementation
        over the non-core points.

        This can also be thought of as a flat clustering derived from constant
        height cut through the single linkage tree.

        This represents the result of selecting a cut value for robust single linkage
        clustering. The `min_cluster_size` allows the flat clustering to declare noise
        points (and cluster smaller than `min_cluster_size`).

        Parameters
        ----------
        cut_distance : float
            The mutual reachability distance cut value to use to generate a
            flat clustering.

        min_cluster_size : int, default=5
            Clusters smaller than this value with be called 'noise' and remain
            unclustered in the resulting flat clustering.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            An array of cluster labels, one per datapoint.
            Outliers are labeled as follows:

            - Noisy samples are given the label -1.
            - Samples with infinite elements (+/- np.inf) are given the label -2.
            - Samples with missing data are given the label -3, even if they
              also have infinite elements.
        