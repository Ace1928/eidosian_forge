import functools
import sys
import math
import warnings
import numpy as np
from .._utils import set_module
import numpy.core.numeric as _nx
from numpy.core.numeric import ScalarType, array
from numpy.core.numerictypes import issubdtype
import numpy.matrixlib as matrixlib
from .function_base import diff
from numpy.core.multiarray import ravel_multi_index, unravel_index
from numpy.core import overrides, linspace
from numpy.lib.stride_tricks import as_strided
class OGridClass(nd_grid):
    """
    An instance which returns an open multi-dimensional "meshgrid".

    An instance which returns an open (i.e. not fleshed out) mesh-grid
    when indexed, so that only one dimension of each returned array is
    greater than 1.  The dimension and number of the output arrays are
    equal to the number of indexing dimensions.  If the step length is
    not a complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid
        `ndarrays` with only one dimension not equal to 1

    See Also
    --------
    mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> from numpy import ogrid
    >>> ogrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> ogrid[0:5,0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]

    """

    def __init__(self):
        super().__init__(sparse=True)