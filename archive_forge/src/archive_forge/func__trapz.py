import numpy as np
from .quantity import Quantity
from .units import dimensionless, radian, degree  # type: ignore[no-redef]
from .decorators import with_doc
def _trapz(y, x, dx, axis):
    """ported from numpy 1.26 since it will be deprecated and removed"""
    try:
        from scipy.integrate import trapezoid
    except ImportError:
        from numpy.core.numeric import asanyarray
        from numpy.core.umath import add
        y = asanyarray(y)
        if x is None:
            d = dx
        else:
            x = asanyarray(x)
            if x.ndim == 1:
                d = diff(x)
                shape = [1] * y.ndim
                shape[axis] = d.shape[0]
                d = d.reshape(shape)
            else:
                d = diff(x, axis=axis)
        nd = y.ndim
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        try:
            ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
        except ValueError:
            d = np.asarray(d)
            y = np.asarray(y)
            ret = add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
        return ret
    else:
        return trapezoid(y, x=x, dx=dx, axis=axis)