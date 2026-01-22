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
class TransformedPath(TransformNode):
    """
    A `TransformedPath` caches a non-affine transformed copy of the
    `~.path.Path`.  This cached copy is automatically updated when the
    non-affine part of the transform changes.

    .. note::

        Paths are considered immutable by this class. Any update to the
        path's vertices/codes will not trigger a transform recomputation.

    """

    def __init__(self, path, transform):
        """
        Parameters
        ----------
        path : `~.path.Path`
        transform : `Transform`
        """
        _api.check_isinstance(Transform, transform=transform)
        super().__init__()
        self._path = path
        self._transform = transform
        self.set_children(transform)
        self._transformed_path = None
        self._transformed_points = None

    def _revalidate(self):
        if self._invalid == self._INVALID_FULL or self._transformed_path is None:
            self._transformed_path = self._transform.transform_path_non_affine(self._path)
            self._transformed_points = Path._fast_from_codes_and_verts(self._transform.transform_non_affine(self._path.vertices), None, self._path)
        self._invalid = 0

    def get_transformed_points_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.  Unlike
        :meth:`get_transformed_path_and_affine`, no interpolation will
        be performed.
        """
        self._revalidate()
        return (self._transformed_points, self.get_affine())

    def get_transformed_path_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.
        """
        self._revalidate()
        return (self._transformed_path, self.get_affine())

    def get_fully_transformed_path(self):
        """
        Return a fully-transformed copy of the child path.
        """
        self._revalidate()
        return self._transform.transform_path_affine(self._transformed_path)

    def get_affine(self):
        return self._transform.get_affine()