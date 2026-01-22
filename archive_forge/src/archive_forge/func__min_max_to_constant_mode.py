import warnings
import numpy as np
from scipy import ndimage as ndi
from .footprints import _footprint_is_sequence, mirror_footprint, pad_footprint
from .misc import default_footprint
from .._shared.utils import DEPRECATED
def _min_max_to_constant_mode(dtype, mode, cval):
    """Replace 'max' and 'min' with appropriate 'cval' and 'constant' mode."""
    if mode == 'max':
        mode = 'constant'
        if np.issubdtype(dtype, bool):
            cval = True
        elif np.issubdtype(dtype, np.integer):
            cval = np.iinfo(dtype).max
        else:
            cval = np.inf
    elif mode == 'min':
        mode = 'constant'
        if np.issubdtype(dtype, bool):
            cval = False
        elif np.issubdtype(dtype, np.integer):
            cval = np.iinfo(dtype).min
        else:
            cval = -np.inf
    return (mode, cval)