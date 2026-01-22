import numpy as np
from ._miobase import convert_dtypes
class MatlabFunction(np.ndarray):
    """Subclass for a MATLAB function.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj