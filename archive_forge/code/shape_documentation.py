import numpy
import cupy
Returns a flattened array.

    It tries to return a view if possible, otherwise returns a copy.

    Args:
        a (cupy.ndarray): Array to be flattened.
        order ({'C', 'F', 'A', 'K'}):
            The elements of ``a`` are read using this index order. 'C' means
            to index the elements in row-major, C-style order,
            with the last axis index changing fastest, back to the first
            axis index changing slowest.  'F' means to index the elements
            in column-major, Fortran-style order, with the
            first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of
            the memory layout of the underlying array, and only refer to
            the order of axis indexing.  'A' means to read the elements in
            Fortran-like index order if ``a`` is Fortran *contiguous* in
            memory, C-like order otherwise.  'K' means to read the
            elements in the order they occur in memory, except for
            reversing the data when strides are negative.  By default, 'C'
            index order is used.

    Returns:
        cupy.ndarray: A flattened view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.ravel`

    