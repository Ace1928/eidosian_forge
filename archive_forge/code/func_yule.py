import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def yule(u, v, w=None):
    """
    Compute the Yule dissimilarity between two boolean 1-D arrays.

    The Yule dissimilarity is defined as

    .. math::

         \\frac{R}{c_{TT} * c_{FF} + \\frac{R}{2}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n` and :math:`R = 2.0 * c_{TF} * c_{FT}`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    yule : double
        The Yule dissimilarity between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.yule([1, 0, 0], [0, 1, 0])
    2.0
    >>> distance.yule([1, 1, 0], [0, 1, 0])
    0.0

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if w is not None:
        w = _validate_weights(w)
    nff, nft, ntf, ntt = _nbool_correspond_all(u, v, w=w)
    half_R = ntf * nft
    if half_R == 0:
        return 0.0
    else:
        return float(2.0 * half_R / (ntt * nff + half_R))