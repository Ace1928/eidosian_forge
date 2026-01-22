import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class _DerivedBivariateSpline(_BivariateSplineBase):
    """Bivariate spline constructed from the coefficients and knots of another
    spline.

    Notes
    -----
    The class is not meant to be instantiated directly from the data to be
    interpolated or smoothed. As a result, its ``fp`` attribute and
    ``get_residual`` method are inherited but overridden; ``AttributeError`` is
    raised when they are accessed.

    The other inherited attributes can be used as usual.
    """
    _invalid_why = 'is unavailable, because _DerivedBivariateSpline instance is not constructed from data that are to be interpolated or smoothed, but derived from the underlying knots and coefficients of another spline object'

    @property
    def fp(self):
        raise AttributeError('attribute "fp" %s' % self._invalid_why)

    def get_residual(self):
        raise AttributeError('method "get_residual" %s' % self._invalid_why)