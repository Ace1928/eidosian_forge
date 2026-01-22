import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def _vode_banded_jac_wrapper(jacfunc, ml, jac_params):
    """
    Wrap a banded Jacobian function with a function that pads
    the Jacobian with `ml` rows of zeros.
    """

    def jac_wrapper(t, y):
        jac = asarray(jacfunc(t, y, *jac_params))
        padded_jac = vstack((jac, zeros((ml, jac.shape[1]))))
        return padded_jac
    return jac_wrapper