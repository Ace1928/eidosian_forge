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
class IdentityTransform(Affine2DBase):
    """
    A special class that does one thing, the identity transform, in a
    fast way.
    """
    _mtx = np.identity(3)

    def frozen(self):
        return self
    __str__ = _make_str_method()

    def get_matrix(self):
        return self._mtx

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform(self, values):
        return np.asanyarray(values)

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform_affine(self, values):
        return np.asanyarray(values)

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform_non_affine(self, values):
        return np.asanyarray(values)

    def transform_path(self, path):
        return path

    def transform_path_affine(self, path):
        return path

    def transform_path_non_affine(self, path):
        return path

    def get_affine(self):
        return self

    def inverted(self):
        return self