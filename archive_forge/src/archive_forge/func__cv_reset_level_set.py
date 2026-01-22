import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_reset_level_set(phi):
    """This is a placeholder function as resetting the level set is not
    strictly necessary, and has not been done for this implementation.
    """
    return phi