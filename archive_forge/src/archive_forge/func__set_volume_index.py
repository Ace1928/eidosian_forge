import weakref
import numpy as np
from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt
def _set_volume_index(self, v, update_slices=True):
    """Set the plot data using a volume index"""
    v = self._data_idx[3] if v is None else int(round(v))
    if v == self._data_idx[3]:
        return
    max_ = np.prod(self._volume_dims)
    self._data_idx[3] = max(min(int(round(v)), max_ - 1), 0)
    idx = (slice(None), slice(None), slice(None))
    if self._data.ndim > 3:
        idx = idx + tuple(np.unravel_index(self._data_idx[3], self._volume_dims))
    self._current_vol_data = self._data[idx]
    if update_slices:
        self._set_position(None, None, None, notify=False)