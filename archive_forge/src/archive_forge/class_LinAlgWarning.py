import numpy as np
from numpy.linalg import LinAlgError
from .blas import get_blas_funcs
from .lapack import get_lapack_funcs
class LinAlgWarning(RuntimeWarning):
    """
    The warning emitted when a linear algebra related operation is close
    to fail conditions of the algorithm or loss of accuracy is expected.
    """
    pass