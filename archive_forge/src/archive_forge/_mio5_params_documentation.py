import numpy as np
from ._miobase import convert_dtypes
Subclass for a MATLAB opaque matrix.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    