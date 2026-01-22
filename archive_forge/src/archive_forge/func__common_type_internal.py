import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def _common_type_internal(default_dtype, *dtypes):
    inexact_dtypes = [dtype if dtype.kind in 'fc' else default_dtype for dtype in dtypes]
    return numpy.result_type(*inexact_dtypes)