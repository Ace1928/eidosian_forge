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
class BlendedGenericTransform(_BlendedMixin, Transform):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This "generic" version can handle any given child transform in the
    *x*- and *y*-directions.
    """
    input_dims = 2
    output_dims = 2
    is_separable = True
    pass_through = True

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to transform the
        *x*-axis and *y_transform* to transform the *y*-axis.

        You will generally not call this constructor directly but use the
        `blended_transform_factory` function instead, which can determine
        automatically which kind of blended transform to create.
        """
        Transform.__init__(self, **kwargs)
        self._x = x_transform
        self._y = y_transform
        self.set_children(x_transform, y_transform)
        self._affine = None

    @property
    def depth(self):
        return max(self._x.depth, self._y.depth)

    def contains_branch(self, other):
        return False
    is_affine = property(lambda self: self._x.is_affine and self._y.is_affine)
    has_inverse = property(lambda self: self._x.has_inverse and self._y.has_inverse)

    def frozen(self):
        return blended_transform_factory(self._x.frozen(), self._y.frozen())

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform_non_affine(self, values):
        if self._x.is_affine and self._y.is_affine:
            return values
        x = self._x
        y = self._y
        if x == y and x.input_dims == 2:
            return x.transform_non_affine(values)
        if x.input_dims == 2:
            x_points = x.transform_non_affine(values)[:, 0:1]
        else:
            x_points = x.transform_non_affine(values[:, 0])
            x_points = x_points.reshape((len(x_points), 1))
        if y.input_dims == 2:
            y_points = y.transform_non_affine(values)[:, 1:]
        else:
            y_points = y.transform_non_affine(values[:, 1])
            y_points = y_points.reshape((len(y_points), 1))
        if isinstance(x_points, np.ma.MaskedArray) or isinstance(y_points, np.ma.MaskedArray):
            return np.ma.concatenate((x_points, y_points), 1)
        else:
            return np.concatenate((x_points, y_points), 1)

    def inverted(self):
        return BlendedGenericTransform(self._x.inverted(), self._y.inverted())

    def get_affine(self):
        if self._invalid or self._affine is None:
            if self._x == self._y:
                self._affine = self._x.get_affine()
            else:
                x_mtx = self._x.get_affine().get_matrix()
                y_mtx = self._y.get_affine().get_matrix()
                mtx = np.array([x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]])
                self._affine = Affine2D(mtx)
            self._invalid = 0
        return self._affine