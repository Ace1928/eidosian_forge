from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def _get_index_dtype(self, arrays=(), maxval=None, check_contents=False):
    """
        Determine index dtype for array.

        This wraps _sputils.get_index_dtype, providing compatibility for both
        array and matrix API sparse matrices. Matrix API sparse matrices would
        attempt to downcast the indices - which can be computationally
        expensive and undesirable for users. The array API changes this
        behaviour.

        See discussion: https://github.com/scipy/scipy/issues/16774

        The get_index_dtype import is due to implementation details of the test
        suite. It allows the decorator ``with_64bit_maxval_limit`` to mock a
        lower int32 max value for checks on the matrix API's downcasting
        behaviour.
        """
    from ._sputils import get_index_dtype
    return get_index_dtype(arrays, maxval, check_contents and (not isinstance(self, sparray)))