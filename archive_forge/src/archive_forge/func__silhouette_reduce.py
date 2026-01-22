import functools
from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_random_state, check_X_y
from ...utils._param_validation import (
from ..pairwise import _VALID_METRICS, pairwise_distances, pairwise_distances_chunked
def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X.

    Parameters
    ----------
    D_chunk : {array-like, sparse matrix} of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk. If a sparse matrix is provided,
        only CSR format is accepted.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    n_chunk_samples = D_chunk.shape[0]
    cluster_distances = np.zeros((n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype)
    if issparse(D_chunk):
        if D_chunk.format != 'csr':
            raise TypeError('Expected CSR matrix. Please pass sparse matrix in CSR format.')
        for i in range(n_chunk_samples):
            indptr = D_chunk.indptr
            indices = D_chunk.indices[indptr[i]:indptr[i + 1]]
            sample_weights = D_chunk.data[indptr[i]:indptr[i + 1]]
            sample_labels = np.take(labels, indices)
            cluster_distances[i] += np.bincount(sample_labels, weights=sample_weights, minlength=len(label_freqs))
    else:
        for i in range(n_chunk_samples):
            sample_weights = D_chunk[i]
            sample_labels = labels
            cluster_distances[i] += np.bincount(sample_labels, weights=sample_weights, minlength=len(label_freqs))
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    intra_cluster_distances = cluster_distances[intra_index]
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)
    return (intra_cluster_distances, inter_cluster_distances)