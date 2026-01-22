import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
    """
    sort + indexing median, faster for small medians along multiple
    dimensions due to the high overhead of apply_along_axis

    see nanmedian for parameter usage
    """
    a = np.ma.masked_array(a, np.isnan(a))
    m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn('All-NaN slice encountered', RuntimeWarning, stacklevel=5)
    fill_value = np.timedelta64('NaT') if m.dtype.kind == 'm' else np.nan
    if out is not None:
        out[...] = m.filled(fill_value)
        return out
    return m.filled(fill_value)