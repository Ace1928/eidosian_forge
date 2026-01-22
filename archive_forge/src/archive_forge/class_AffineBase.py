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
class AffineBase(Transform):
    """
    The base class of all affine transformations of any number of dimensions.
    """
    is_affine = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inverted = None

    def __array__(self, *args, **kwargs):
        return self.get_matrix()

    def __eq__(self, other):
        if getattr(other, 'is_affine', False) and hasattr(other, 'get_matrix'):
            return (self.get_matrix() == other.get_matrix()).all()
        return NotImplemented

    def transform(self, values):
        return self.transform_affine(values)

    def transform_affine(self, values):
        raise NotImplementedError('Affine subclasses should override this method.')

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform_non_affine(self, values):
        return values

    def transform_path(self, path):
        return self.transform_path_affine(path)

    def transform_path_affine(self, path):
        return Path(self.transform_affine(path.vertices), path.codes, path._interpolation_steps)

    def transform_path_non_affine(self, path):
        return path

    def get_affine(self):
        return self