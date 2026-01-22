import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
def _set_class(self, cls):
    self._spline_class = cls
    if self.__class__ in (UnivariateSpline, InterpolatedUnivariateSpline, LSQUnivariateSpline):
        self.__class__ = cls
    else:
        pass