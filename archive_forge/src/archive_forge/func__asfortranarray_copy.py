import scipy.linalg._interpolative as _id
import numpy as np
def _asfortranarray_copy(A):
    """
    Same as np.asfortranarray, but ensure a copy
    """
    A = np.asarray(A)
    if A.flags.f_contiguous:
        A = A.copy(order='F')
    else:
        A = np.asfortranarray(A)
    return A