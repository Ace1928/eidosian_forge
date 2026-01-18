from numbers import Integral, Real
from time import time
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from ..base import (
from ..decomposition import PCA
from ..metrics.pairwise import _VALID_METRICS, pairwise_distances
from ..neighbors import NearestNeighbors
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _num_samples, check_non_negative
from . import _barnes_hut_tsne, _utils  # type: ignore
@validate_params({'X': ['array-like', 'sparse matrix'], 'X_embedded': ['array-like', 'sparse matrix'], 'n_neighbors': [Interval(Integral, 1, None, closed='left')], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable]}, prefer_skip_nested_validation=True)
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric='euclidean'):
    """Indicate to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \\frac{2}{nk (2n - 3k - 1)} \\sum^n_{i=1}
            \\sum_{j \\in \\mathcal{N}_{i}^{k}} \\max(0, (r(i, j) - k))

    where for each sample i, :math:`\\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or \\
        (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : {array-like, sparse matrix} of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

        .. versionadded:: 0.20

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelfth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.manifold import trustworthiness
    >>> X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    >>> X_embedded = PCA(n_components=2).fit_transform(X)
    >>> print(f"{trustworthiness(X, X_embedded, n_neighbors=5):.2f}")
    0.92
    """
    n_samples = _num_samples(X)
    if n_neighbors >= n_samples / 2:
        raise ValueError(f'n_neighbors ({n_neighbors}) should be less than n_samples / 2 ({n_samples / 2})')
    dist_X = pairwise_distances(X, metric=metric)
    if metric == 'precomputed':
        dist_X = dist_X.copy()
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embedded).kneighbors(return_distance=False)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
    ranks = inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t