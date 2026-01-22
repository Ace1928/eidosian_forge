import warnings
from heapq import heapify, heappop, heappush, heappushpop
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from ..base import (
from ..metrics import DistanceMetric
from ..metrics._dist_metrics import METRIC_MAPPING64
from ..metrics.pairwise import _VALID_METRICS, paired_distances
from ..utils import check_array
from ..utils._fast_dict import IntFloatDict
from ..utils._param_validation import (
from ..utils.graph import _fix_connected_components
from ..utils.validation import check_memory
from . import _hierarchical_fast as _hierarchical  # type: ignore
from ._feature_agglomeration import AgglomerationTransform
class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    """
    Agglomerative Clustering.

    Recursively merges pair of clusters of sample data; uses linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted. If "precomputed", a distance matrix is needed
        as input for the fit method.

        .. versionadded:: 1.2

        .. deprecated:: 1.4
           `metric=None` is deprecated in 1.4 and will be removed in 1.6.
           Let `metric` be the default value (i.e. `"euclidean"`) instead.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - 'ward' minimizes the variance of the clusters being merged.
        - 'average' uses the average of the distances of each observation of
          the two sets.
        - 'complete' or 'maximum' linkage uses the maximum distances between
          all observations of the two sets.
        - 'single' uses the minimum of the distances between all observations
          of the two sets.

        .. versionadded:: 0.20
            Added the 'single' option

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    FeatureAgglomeration : Agglomerative clustering but for features instead of
        samples.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering
    AgglomerativeClustering()
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    """
    _parameter_constraints: dict = {'n_clusters': [Interval(Integral, 1, None, closed='left'), None], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable, Hidden(None)], 'memory': [str, HasMethods('cache'), None], 'connectivity': ['array-like', callable, None], 'compute_full_tree': [StrOptions({'auto'}), 'boolean'], 'linkage': [StrOptions(set(_TREE_BUILDERS.keys()))], 'distance_threshold': [Interval(Real, 0, None, closed='left'), None], 'compute_distances': ['boolean']}

    def __init__(self, n_clusters=2, *, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.metric = metric
        self.compute_distances = compute_distances

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the hierarchical clustering from features, or distance matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or                 (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        X = self._validate_data(X, ensure_min_samples=2)
        return self._fit(X)

    def _fit(self, X):
        """Fit without validation

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        memory = check_memory(self.memory)
        if self.metric is None:
            warnings.warn("`metric=None` is deprecated in version 1.4 and will be removed in version 1.6. Let `metric` be the default value (i.e. `'euclidean'`) instead.", FutureWarning)
            self._metric = 'euclidean'
        else:
            self._metric = self.metric
        if not (self.n_clusters is None) ^ (self.distance_threshold is None):
            raise ValueError('Exactly one of n_clusters and distance_threshold has to be set, and the other needs to be None.')
        if self.distance_threshold is not None and (not self.compute_full_tree):
            raise ValueError('compute_full_tree must be True if distance_threshold is set.')
        if self.linkage == 'ward' and self._metric != 'euclidean':
            raise ValueError(f'{self._metric} was provided as metric. Ward can only work with euclidean distances.')
        tree_builder = _TREE_BUILDERS[self.linkage]
        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(connectivity, accept_sparse=['csr', 'coo', 'lil'])
        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            if self.distance_threshold is not None:
                compute_full_tree = True
            else:
                compute_full_tree = self.n_clusters < max(100, 0.02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None
        kwargs = {}
        if self.linkage != 'ward':
            kwargs['linkage'] = self.linkage
            kwargs['affinity'] = self._metric
        distance_threshold = self.distance_threshold
        return_distance = distance_threshold is not None or self.compute_distances
        out = memory.cache(tree_builder)(X, connectivity=connectivity, n_clusters=n_clusters, return_distance=return_distance, **kwargs)
        self.children_, self.n_connected_components_, self.n_leaves_, parents = out[:4]
        if return_distance:
            self.distances_ = out[-1]
        if self.distance_threshold is not None:
            self.n_clusters_ = np.count_nonzero(self.distances_ >= distance_threshold) + 1
        else:
            self.n_clusters_ = self.n_clusters
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters_, self.children_, self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            labels = np.copy(labels[:n_samples])
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self

    def fit_predict(self, X, y=None):
        """Fit and return the result of each sample's clustering assignment.

        In addition to fitting, this method also return the result of the
        clustering assignment for each sample in the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or                 (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)