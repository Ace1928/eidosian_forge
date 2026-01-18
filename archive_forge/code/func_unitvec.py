from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
def unitvec(vec, norm='l2', return_norm=False):
    """Scale a vector to unit length.

    Parameters
    ----------
    vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
        Input vector in any format
    norm : {'l1', 'l2', 'unique'}, optional
        Metric to normalize in.
    return_norm : bool, optional
        Return the length of vector `vec`, in addition to the normalized vector itself?

    Returns
    -------
    numpy.ndarray, scipy.sparse, list of (int, float)}
        Normalized vector in same format as `vec`.
    float
        Length of `vec` before normalization, if `return_norm` is set.

    Notes
    -----
    Zero-vector will be unchanged.

    """
    supported_norms = ('l1', 'l2', 'unique')
    if norm not in supported_norms:
        raise ValueError("'%s' is not a supported norm. Currently supported norms are %s." % (norm, supported_norms))
    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if norm == 'unique':
            veclen = vec.nnz
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            vec /= veclen
            if return_norm:
                return (vec, veclen)
            else:
                return vec
        elif return_norm:
            return (vec, 1.0)
        else:
            return vec
    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            if vec.size == 0:
                veclen = 0.0
            else:
                veclen = blas_nrm2(vec)
        if norm == 'unique':
            veclen = np.count_nonzero(vec)
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            if return_norm:
                return (blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen)
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        elif return_norm:
            return (vec, 1.0)
        else:
            return vec
    try:
        first = next(iter(vec))
    except StopIteration:
        if return_norm:
            return (vec, 1.0)
        else:
            return vec
    if isinstance(first, (tuple, list)) and len(first) == 2:
        if norm == 'l1':
            length = float(sum((abs(val) for _, val in vec)))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum((val ** 2 for _, val in vec)))
        if norm == 'unique':
            length = 1.0 * len(vec)
        assert length > 0.0, 'sparse documents must not contain any explicit zero entries'
        if return_norm:
            return (ret_normalized_vec(vec, length), length)
        else:
            return ret_normalized_vec(vec, length)
    else:
        raise ValueError('unknown input type')