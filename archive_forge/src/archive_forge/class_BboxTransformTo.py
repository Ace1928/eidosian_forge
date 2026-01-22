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
class BboxTransformTo(Affine2DBase):
    """
    `BboxTransformTo` is a transformation that linearly transforms points from
    the unit bounding box to a given `Bbox`.
    """
    is_separable = True

    def __init__(self, boxout, **kwargs):
        """
        Create a new `BboxTransformTo` that linearly transforms
        points from the unit bounding box to *boxout*.
        """
        if not boxout.is_bbox:
            raise ValueError("'boxout' must be bbox")
        super().__init__(**kwargs)
        self._boxout = boxout
        self.set_children(boxout)
        self._mtx = None
        self._inverted = None
    __str__ = _make_str_method('_boxout')

    def get_matrix(self):
        if self._invalid:
            outl, outb, outw, outh = self._boxout.bounds
            if DEBUG and (outw == 0 or outh == 0):
                raise ValueError('Transforming to a singular bounding box.')
            self._mtx = np.array([[outw, 0.0, outl], [0.0, outh, outb], [0.0, 0.0, 1.0]], float)
            self._inverted = None
            self._invalid = 0
        return self._mtx