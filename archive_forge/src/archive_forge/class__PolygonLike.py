from __future__ import annotations
from math import floor
import numpy as np
from toolz import memoize
from datashader.glyphs.points import _PointLike
from datashader.utils import isreal, ngjit
class _PolygonLike(_PointLike):
    """_PointLike class, with methods overridden for vertex-delimited shapes.

    Key differences from _PointLike:
        - added self.z as a list, representing vertex weights
        - constructor accepts additional kwargs:
            * weight_type (bool): Whether the weights are on vertices (True) or on the shapes
                                  (False)
            * interp (bool): Whether to interpolate (True), or to have one color per shape (False)
    """

    def __init__(self, x, y, z=None, weight_type=True, interp=True):
        super(_PolygonLike, self).__init__(x, y)
        if z is None:
            self.z = []
        else:
            self.z = z
        self.interpolate = interp
        self.weight_type = weight_type

    @property
    def ndims(self):
        return None

    @property
    def inputs(self):
        return tuple([self.x, self.y] + list(self.z)) + (self.weight_type, self.interpolate)

    def validate(self, in_dshape):
        for col in [self.x, self.y] + list(self.z):
            if not isreal(in_dshape.measure[str(col)]):
                raise ValueError('{} must be real'.format(col))

    def required_columns(self):
        return [self.x, self.y] + list(self.z)

    def compute_x_bounds(self, df):
        xs = df[self.x].values
        bounds = self._compute_bounds(xs.reshape(np.prod(xs.shape)))
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        ys = df[self.y].values
        bounds = self._compute_bounds(ys.reshape(np.prod(ys.shape)))
        return self.maybe_expand_bounds(bounds)