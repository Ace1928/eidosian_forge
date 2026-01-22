import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
class Affine2DBase(AffineBase):
    """
    The base class of all 2D affine transformations.

    2D affine transformations are performed using a 3x3 numpy array::

        a c e
        b d f
        0 0 1

    This class provides the read-only interface.  For a mutable 2D
    affine transformation, use `Affine2D`.

    Subclasses of this class will generally only need to override a
    constructor and `~.Transform.get_matrix` that generates a custom 3x3 matrix.
    """
    input_dims = 2
    output_dims = 2

    def frozen(self):
        return Affine2D(self.get_matrix().copy())

    @property
    def is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == mtx[1, 0] == 0.0

    def to_values(self):
        """
        Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
        """
        mtx = self.get_matrix()
        return tuple(mtx[:2].swapaxes(0, 1).flat)

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform_affine(self, values):
        mtx = self.get_matrix()
        if isinstance(values, np.ma.MaskedArray):
            tpoints = affine_transform(values.data, mtx)
            return np.ma.MaskedArray(tpoints, mask=np.ma.getmask(values))
        return affine_transform(values, mtx)
    if DEBUG:
        _transform_affine = transform_affine

        @_api.rename_parameter('3.8', 'points', 'values')
        def transform_affine(self, values):
            if not isinstance(values, np.ndarray):
                _api.warn_external(f'A non-numpy array of type {type(values)} was passed in for transformation, which results in poor performance.')
            return self._transform_affine(values)

    def inverted(self):
        if self._inverted is None or self._invalid:
            mtx = self.get_matrix()
            shorthand_name = None
            if self._shorthand_name:
                shorthand_name = '(%s)-1' % self._shorthand_name
            self._inverted = Affine2D(inv(mtx), shorthand_name=shorthand_name)
            self._invalid = 0
        return self._inverted