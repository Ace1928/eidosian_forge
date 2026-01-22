import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import _euclidean_distances, euclidean_distances
from ..utils import check_array, check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_info, threadpool_limits
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.validation import (
from ._k_means_common import (
from ._k_means_elkan import (
from ._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
from ._k_means_minibatch import _minibatch_update_dense, _minibatch_update_sparse
def _mini_batch_step(X, sample_weight, centers, centers_new, weight_sums, random_state, random_reassign=False, reassignment_ratio=0.01, verbose=False, n_threads=1):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The original data array. If sparse, must be in CSR format.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared euclidean norm of each data point.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        The inertia is computed after finding the labels and before updating
        the centers.
    """
    labels, inertia = _labels_inertia(X, sample_weight, centers, n_threads=n_threads)
    if sp.issparse(X):
        _minibatch_update_sparse(X, sample_weight, centers, centers_new, weight_sums, labels, n_threads)
    else:
        _minibatch_update_dense(X, sample_weight, centers, centers_new, weight_sums, labels, n_threads)
    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()
        if to_reassign.sum() > 0.5 * X.shape[0]:
            indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()
        if n_reassigns:
            new_centers = random_state.choice(X.shape[0], replace=False, size=n_reassigns)
            if verbose:
                print(f'[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.')
            if sp.issparse(X):
                assign_rows_csr(X, new_centers.astype(np.intp, copy=False), np.where(to_reassign)[0].astype(np.intp, copy=False), centers_new)
            else:
                centers_new[to_reassign] = X[new_centers]
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])
    return inertia