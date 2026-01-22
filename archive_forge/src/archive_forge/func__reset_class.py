import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
def _reset_class(self):
    data = self._data
    n, t, c, k, ier = (data[7], data[8], data[9], data[5], data[-1])
    self._eval_args = (t[:n], c[:n], k)
    if ier == 0:
        pass
    elif ier == -1:
        self._set_class(InterpolatedUnivariateSpline)
    elif ier == -2:
        self._set_class(LSQUnivariateSpline)
    else:
        if ier == 1:
            self._set_class(LSQUnivariateSpline)
        message = _curfit_messages.get(ier, 'ier=%s' % ier)
        warnings.warn(message, stacklevel=3)