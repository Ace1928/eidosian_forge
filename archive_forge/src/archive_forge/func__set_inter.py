import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def _set_inter(self, val):
    self._inter = np.squeeze(self.scaler_dtype.type(val))