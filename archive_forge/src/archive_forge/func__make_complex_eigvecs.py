import warnings
import numpy
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def _make_complex_eigvecs(w, vin, dtype):
    """
    Produce complex-valued eigenvectors from LAPACK DGGEV real-valued output
    """
    v = numpy.array(vin, dtype=dtype)
    m = w.imag > 0
    m[:-1] |= w.imag[1:] < 0
    for i in flatnonzero(m):
        v.imag[:, i] = vin[:, i + 1]
        conj(v[:, i], v[:, i + 1])
    return v