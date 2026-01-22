import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_delta(x, eps=1.0):
    """Returns the result of a regularised dirac function of the
    input value(s).
    """
    return eps / (eps ** 2 + x ** 2)