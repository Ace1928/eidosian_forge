from __future__ import annotations
from xarray.core import dtypes, nputils
def _fill_with_last_one(a, b):
    return np.where(~np.isnan(b), b, a)