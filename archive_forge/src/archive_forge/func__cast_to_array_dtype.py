import warnings
from scipy.linalg import qr as s_qr
from scipy import integrate, interpolate, linalg
from scipy.interpolate import make_interp_spline
from ._filter_design import (tf2zpk, zpk2tf, normalize, freqs, freqz, freqs_zpk,
from ._lti_conversion import (tf2ss, abcd_normalize, ss2tf, zpk2ss, ss2zpk,
import numpy
import numpy as np
from numpy import (real, atleast_1d, squeeze, asarray, zeros,
import copy
def _cast_to_array_dtype(in1, in2):
    """Cast array to dtype of other array, while avoiding ComplexWarning.

    Those can be raised when casting complex to real.
    """
    if numpy.issubdtype(in2.dtype, numpy.float64):
        in1 = in1.real.astype(in2.dtype)
    else:
        in1 = in1.astype(in2.dtype)
    return in1