import warnings
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics import euclidean_distances, pairwise_distances_argmin
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import check_is_fitted
@validate_params({'S': ['array-like'], 'return_n_iter': ['boolean']}, prefer_skip_nested_validation=False)
def affinity_propagation(S, *, preference=None, convergence_iter=15, max_iter=200, damping=0.5, copy=True, verbose=False, return_n_iter=False, random_state=None):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    S : array-like of shape (n_samples, n_samples)
        Matrix of similarities between points.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, default=200
        Maximum number of iterations.

    damping : float, default=0.5
        Damping factor between 0.5 and 1.

    copy : bool, default=True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency.

    verbose : bool, default=False
        The verbosity level.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Returns
    -------
    cluster_centers_indices : ndarray of shape (n_clusters,)
        Index of clusters centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it will still return a arrays of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import affinity_propagation
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> S = -euclidean_distances(X, squared=True)
    >>> cluster_centers_indices, labels = affinity_propagation(S, random_state=0)
    >>> cluster_centers_indices
    array([0, 3])
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    """
    estimator = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, copy=copy, preference=preference, affinity='precomputed', verbose=verbose, random_state=random_state).fit(S)
    if return_n_iter:
        return (estimator.cluster_centers_indices_, estimator.labels_, estimator.n_iter_)
    return (estimator.cluster_centers_indices_, estimator.labels_)