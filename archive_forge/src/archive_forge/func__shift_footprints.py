import warnings
import numpy as np
from scipy import ndimage as ndi
from .footprints import _footprint_is_sequence, mirror_footprint, pad_footprint
from .misc import default_footprint
from .._shared.utils import DEPRECATED
def _shift_footprints(footprint, shift_x, shift_y):
    """Shifts the footprints, whether it's a single array or a sequence.

    See `_shift_footprint`, which is called for each array in the sequence.
    """
    if shift_x is DEPRECATED and shift_y is DEPRECATED:
        return footprint
    warning_msg = 'The parameters `shift_x` and `shift_y` are deprecated since v0.23 and will be removed in v0.26. Use `pad_footprint` or modify the footprintmanually instead.'
    warnings.warn(warning_msg, FutureWarning, stacklevel=4)
    if _footprint_is_sequence(footprint):
        return tuple(((_shift_footprint(fp, shift_x, shift_y), n) for fp, n in footprint))
    return _shift_footprint(footprint, shift_x, shift_y)