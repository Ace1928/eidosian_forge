from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def working_type(in_type: npt.DTypeLike, slope: npt.ArrayLike=1.0, inter: npt.ArrayLike=0.0) -> type[np.number]:
    """Return array type from applying `slope`, `inter` to array of `in_type`

    Numpy type that results from an array of type `in_type` being combined with
    `slope` and `inter`. It returns something like the dtype type of
    ``((np.zeros((2,), dtype=in_type) - inter) / slope)``, but ignoring the
    actual values of `slope` and `inter`.

    Note that you would not necessarily get the same type by applying slope and
    inter the other way round.  Also, you'll see that the order in which slope
    and inter are applied is the opposite of the order in which they are
    passed.

    Parameters
    ----------
    in_type : numpy type specifier
        Numpy type of input array.  Any valid input for ``np.dtype()``
    slope : scalar, optional
        slope to apply to array.  If 1.0 (default), ignore this value and its
        type.
    inter : scalar, optional
        intercept to apply to array.  If 0.0 (default), ignore this value and
        its type.

    Returns
    -------
    wtype: numpy type
        Numpy type resulting from applying `inter` and `slope` to array of type
        `in_type`.
    """
    val = np.array([1], dtype=in_type)
    if inter != 0:
        val = val + np.array([0], dtype=np.array(inter).dtype)
    if slope != 1:
        val = val / np.array([1], dtype=np.array(slope).dtype)
    return val.dtype.type