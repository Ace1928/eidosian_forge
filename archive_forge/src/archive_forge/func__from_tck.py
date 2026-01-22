import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
@classmethod
def _from_tck(cls, tck):
    """Construct a spline object from given tck and degree"""
    self = cls.__new__(cls)
    if len(tck) != 5:
        raise ValueError('tck should be a 5 element tuple of tx, ty, c, kx, ky')
    self.tck = tck[:3]
    self.degrees = tck[3:]
    return self