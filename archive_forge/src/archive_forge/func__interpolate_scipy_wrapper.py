from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def _interpolate_scipy_wrapper(x: np.ndarray, y: np.ndarray, new_x: np.ndarray, method: str, fill_value=None, bounds_error: bool=False, order=None, **kwargs):
    """
    Passed off to scipy.interpolate.interp1d. method is scipy's kind.
    Returns an array interpolated at new_x.  Add any new methods to
    the list in _clean_interp_method.
    """
    extra = f'{method} interpolation requires SciPy.'
    import_optional_dependency('scipy', extra=extra)
    from scipy import interpolate
    new_x = np.asarray(new_x)
    alt_methods = {'barycentric': interpolate.barycentric_interpolate, 'krogh': interpolate.krogh_interpolate, 'from_derivatives': _from_derivatives, 'piecewise_polynomial': _from_derivatives, 'cubicspline': _cubicspline_interpolate, 'akima': _akima_interpolate, 'pchip': interpolate.pchip_interpolate}
    interp1d_methods = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial']
    if method in interp1d_methods:
        if method == 'polynomial':
            kind = order
        else:
            kind = method
        terp = interpolate.interp1d(x, y, kind=kind, fill_value=fill_value, bounds_error=bounds_error)
        new_y = terp(new_x)
    elif method == 'spline':
        if isna(order) or order <= 0:
            raise ValueError(f'order needs to be specified and greater than 0; got order: {order}')
        terp = interpolate.UnivariateSpline(x, y, k=order, **kwargs)
        new_y = terp(new_x)
    else:
        if not x.flags.writeable:
            x = x.copy()
        if not y.flags.writeable:
            y = y.copy()
        if not new_x.flags.writeable:
            new_x = new_x.copy()
        terp = alt_methods[method]
        new_y = terp(x, y, new_x, **kwargs)
    return new_y