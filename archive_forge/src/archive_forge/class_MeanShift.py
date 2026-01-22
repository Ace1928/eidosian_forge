import warnings
from collections import defaultdict
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import pairwise_distances_argmin
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state, gen_batches
from ..utils._param_validation import Interval, validate_params
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
class MeanShift(ClusterMixin, BaseEstimator):
    """Mean shift clustering using a flat kernel.

    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.

    Seeding is performed using a binning technique for scalability.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------
    bandwidth : float, default=None
        Bandwidth used in the flat kernel.

        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).

    seeds : array-like of shape (n_samples, n_features), default=None
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.

    bin_seeding : bool, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        The default value is False.
        Ignored if seeds argument is not None.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    n_jobs : int, default=None
        The number of jobs to use for the computation. The following tasks benefit
        from the parallelization:

        - The search of nearest neighbors for bandwidth estimation and label
          assignments. See the details in the docstring of the
          ``NearestNeighbors`` class.
        - Hill-climbing optimization for all seeds.

        See :term:`Glossary <n_jobs>` for more details.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

        .. versionadded:: 0.22

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    n_iter_ : int
        Maximum number of iterations performed on each seed.

        .. versionadded:: 0.22

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : K-Means clustering.

    Notes
    -----

    Scalability:

    Because this implementation uses a flat kernel and
    a Ball Tree to look up members of each kernel, the complexity will tend
    towards O(T*n*log(n)) in lower dimensions, with n the number of samples
    and T the number of points. In higher dimensions the complexity will
    tend towards O(T*n^2).

    Scalability can be boosted by using fewer seeds, for example by using
    a higher value of min_bin_freq in the get_bin_seeds function.

    Note that the estimate_bandwidth function is much less scalable than the
    mean shift algorithm and will be the bottleneck if it is used.

    References
    ----------

    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.

    Examples
    --------
    >>> from sklearn.cluster import MeanShift
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = MeanShift(bandwidth=2).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering.predict([[0, 0], [5, 5]])
    array([1, 0])
    >>> clustering
    MeanShift(bandwidth=2)
    """
    _parameter_constraints: dict = {'bandwidth': [Interval(Real, 0, None, closed='neither'), None], 'seeds': ['array-like', None], 'bin_seeding': ['boolean'], 'min_bin_freq': [Interval(Integral, 1, None, closed='left')], 'cluster_all': ['boolean'], 'n_jobs': [Integral, None], 'max_iter': [Interval(Integral, 0, None, closed='left')]}

    def __init__(self, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.min_bin_freq = min_bin_freq
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
               Fitted instance.
        """
        X = self._validate_data(X)
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, n_jobs=self.n_jobs)
        seeds = self.seeds
        if seeds is None:
            if self.bin_seeding:
                seeds = get_bin_seeds(X, bandwidth, self.min_bin_freq)
            else:
                seeds = X
        n_samples, n_features = X.shape
        center_intensity_dict = {}
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
        all_res = Parallel(n_jobs=self.n_jobs)((delayed(_mean_shift_single_seed)(seed, X, nbrs, self.max_iter) for seed in seeds))
        for i in range(len(seeds)):
            if all_res[i][1]:
                center_intensity_dict[all_res[i][0]] = all_res[i][1]
        self.n_iter_ = max([x[2] for x in all_res])
        if not center_intensity_dict:
            raise ValueError('No point was within bandwidth=%f of any seed. Try a different seeding strategy                              or increase the bandwidth.' % bandwidth)
        sorted_by_intensity = sorted(center_intensity_dict.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        unique = np.ones(len(sorted_centers), dtype=bool)
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=self.n_jobs).fit(sorted_centers)
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center], return_distance=False)[0]
                unique[neighbor_idxs] = 0
                unique[i] = 1
        cluster_centers = sorted_centers[unique]
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(cluster_centers)
        labels = np.zeros(n_samples, dtype=int)
        distances, idxs = nbrs.kneighbors(X)
        if self.cluster_all:
            labels = idxs.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels[bool_selector] = idxs.flatten()[bool_selector]
        self.cluster_centers_, self.labels_ = (cluster_centers, labels)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        with config_context(assume_finite=True):
            return pairwise_distances_argmin(X, self.cluster_centers_)