import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def _get_numpy_hist_range(image, source_range):
    if source_range == 'image':
        hist_range = None
    elif source_range == 'dtype':
        hist_range = dtype_limits(image, clip_negative=False)
    else:
        raise ValueError(f'Incorrect value for `source_range` argument: {source_range}')
    return hist_range