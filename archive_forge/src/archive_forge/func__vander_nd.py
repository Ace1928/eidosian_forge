import operator
import functools
import warnings
import numpy as np
from numpy.core.multiarray import dragon4_positional, dragon4_scientific
from numpy.core.umath import absolute
def _vander_nd(vander_fs, points, degrees):
    """
    A generalization of the Vandermonde matrix for N dimensions

    The result is built by combining the results of 1d Vandermonde matrices,

    .. math::
        W[i_0, \\ldots, i_M, j_0, \\ldots, j_N] = \\prod_{k=0}^N{V_k(x_k)[i_0, \\ldots, i_M, j_k]}

    where

    .. math::
        N &= \\texttt{len(points)} = \\texttt{len(degrees)} = \\texttt{len(vander\\_fs)} \\\\
        M &= \\texttt{points[k].ndim} \\\\
        V_k &= \\texttt{vander\\_fs[k]} \\\\
        x_k &= \\texttt{points[k]} \\\\
        0 \\le j_k &\\le \\texttt{degrees[k]}

    Expanding the one-dimensional :math:`V_k` functions gives:

    .. math::
        W[i_0, \\ldots, i_M, j_0, \\ldots, j_N] = \\prod_{k=0}^N{B_{k, j_k}(x_k[i_0, \\ldots, i_M])}

    where :math:`B_{k,m}` is the m'th basis of the polynomial construction used along
    dimension :math:`k`. For a regular polynomial, :math:`B_{k, m}(x) = P_m(x) = x^m`.

    Parameters
    ----------
    vander_fs : Sequence[function(array_like, int) -> ndarray]
        The 1d vander function to use for each axis, such as ``polyvander``
    points : Sequence[array_like]
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
        This must be the same length as `vander_fs`.
    degrees : Sequence[int]
        The maximum degree (inclusive) to use for each axis.
        This must be the same length as `vander_fs`.

    Returns
    -------
    vander_nd : ndarray
        An array of shape ``points[0].shape + tuple(d + 1 for d in degrees)``.
    """
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(f'Expected {n_dims} dimensions of sample points, got {len(points)}')
    if n_dims != len(degrees):
        raise ValueError(f'Expected {n_dims} dimensions of degrees, got {len(degrees)}')
    if n_dims == 0:
        raise ValueError('Unable to guess a dtype or shape when no points are given')
    points = tuple(np.array(tuple(points), copy=False) + 0.0)
    vander_arrays = (vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)] for i in range(n_dims))
    return functools.reduce(operator.mul, vander_arrays)