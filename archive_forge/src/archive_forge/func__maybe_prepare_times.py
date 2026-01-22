from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
def _maybe_prepare_times(var):
    data = var.data
    if data.dtype.kind in 'iu':
        units = var.attrs.get('units', None)
        if units is not None:
            if coding.variables._is_time_like(units):
                mask = data == np.iinfo(np.int64).min
                if mask.any():
                    data = np.where(mask, var.attrs.get('_FillValue', np.nan), data)
    return data