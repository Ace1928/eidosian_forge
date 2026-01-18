import numpy as np
import functools
import operator
A ``einsum`` implementation for ``numpy`` arrays with object dtype.
    The loop is performed in python, meaning the objects themselves need
    only to implement ``__mul__`` and ``__add__`` for the contraction to be
    computed. This may be useful when, for example, computing expressions of
    tensors with symbolic elements, but note it will be very slow when compared
    to ``numpy.einsum`` and numeric data types!

    Parameters
    ----------
    eq : str
        The contraction string, should specify output.
    arrays : sequence of arrays
        These can be any indexable arrays as long as addition and
        multiplication is defined on the elements.

    Returns
    -------
    out : numpy.ndarray
        The output tensor, with ``dtype=object``.
    