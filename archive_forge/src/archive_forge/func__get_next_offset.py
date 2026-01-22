import numbers
from functools import reduce
from operator import mul
import numpy as np
def _get_next_offset(self):
    """Offset in ``self._data`` at which to write next rowelement"""
    if len(self._offsets) == 0:
        return 0
    imax = np.argmax(self._offsets)
    return self._offsets[imax] + self._lengths[imax]