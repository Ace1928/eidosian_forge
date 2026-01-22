import itertools
from ..base import ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
from ..utils._param_validation import (
from ..utils.validation import check_is_fitted
from ._base import VALID_METRICS, KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
from ._unsupervised import NearestNeighbors
class KNeighborsTransformer(ClassNamePrefixFeaturesOutMixin, KNeighborsMixin, TransformerMixin, NeighborsBase):
    """Transform X into a (weighted) graph of k nearest neighbors.

    The transformed data is a sparse graph as returned by kneighbors_graph.

    Read more in the :ref:`User Guide <neighbors_transformer>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    mode : {'distance', 'connectivity'}, default='distance'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    n_neighbors : int, default=5
        Number of neighbors for each sample in the transformed sparse graph.
        For compatibility reasons, as each sample is considered as its own
        neighbor, one extra neighbor will be computed when mode == 'distance'.
        In this case, the sparse graph contains (n_neighbors + 1) neighbors.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        This parameter is expected to be positive.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    kneighbors_graph : Compute the weighted graph of k-neighbors for
        points in X.
    RadiusNeighborsTransformer : Transform X into a weighted graph of
        neighbors nearer than a radius.

    Notes
    -----
    For an example of using :class:`~sklearn.neighbors.KNeighborsTransformer`
    in combination with :class:`~sklearn.manifold.TSNE` see
    :ref:`sphx_glr_auto_examples_neighbors_approximate_nearest_neighbors.py`.

    Examples
    --------
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.neighbors import KNeighborsTransformer
    >>> X, _ = load_wine(return_X_y=True)
    >>> X.shape
    (178, 13)
    >>> transformer = KNeighborsTransformer(n_neighbors=5, mode='distance')
    >>> X_dist_graph = transformer.fit_transform(X)
    >>> X_dist_graph.shape
    (178, 178)
    """
    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints, 'mode': [StrOptions({'distance', 'connectivity'})]}
    _parameter_constraints.pop('radius')

    def __init__(self, *, mode='distance', n_neighbors=5, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None):
        super(KNeighborsTransformer, self).__init__(n_neighbors=n_neighbors, radius=None, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
        self.mode = mode

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        """Fit the k-nearest neighbors transformer from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : KNeighborsTransformer
            The fitted k-nearest neighbors transformer.
        """
        self._fit(X)
        self._n_features_out = self.n_samples_fit_
        return self

    def transform(self, X):
        """Compute the (weighted) graph of Neighbors for points in X.

        Parameters
        ----------
        X : array-like of shape (n_samples_transform, n_features)
            Sample data.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples_transform, n_samples_fit)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        check_is_fitted(self)
        add_one = self.mode == 'distance'
        return self.kneighbors_graph(X, mode=self.mode, n_neighbors=self.n_neighbors + add_one)

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples, n_samples)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        return self.fit(X).transform(X)

    def _more_tags(self):
        return {'_xfail_checks': {'check_methods_sample_order_invariance': 'check is not applicable.'}}