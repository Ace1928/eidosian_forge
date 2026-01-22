import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def _needs_nan2zero(self):
    """True if nan2zero check needed for writing array"""
    return self._nan2zero and self._array.dtype.kind in 'fc' and (self.out_dtype.kind in 'iu') and self.has_nan