from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def unumpy_to_numpy_matrix(arr):
    """
    If arr in a unumpy.matrix, it is converted to a numpy.matrix.
    Otherwise, it is returned unchanged.
    """
    if isinstance(arr, matrix):
        return arr.view(numpy.matrix)
    else:
        return arr