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
class BboxTransform(Affine2DBase):
    """
    `BboxTransform` linearly transforms points from one `Bbox` to another.
    """
    is_separable = True

    def __init__(self, boxin, boxout, **kwargs):
        """
        Create a new `BboxTransform` that linearly transforms
        points from *boxin* to *boxout*.
        """
        if not boxin.is_bbox or not boxout.is_bbox:
            raise ValueError("'boxin' and 'boxout' must be bbox")
        super().__init__(**kwargs)
        self._boxin = boxin
        self._boxout = boxout
        self.set_children(boxin, boxout)
        self._mtx = None
        self._inverted = None
    __str__ = _make_str_method('_boxin', '_boxout')

    def get_matrix(self):
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            outl, outb, outw, outh = self._boxout.bounds
            x_scale = outw / inw
            y_scale = outh / inh
            if DEBUG and (x_scale == 0 or y_scale == 0):
                raise ValueError('Transforming from or to a singular bounding box')
            self._mtx = np.array([[x_scale, 0.0, -inl * x_scale + outl], [0.0, y_scale, -inb * y_scale + outb], [0.0, 0.0, 1.0]], float)
            self._inverted = None
            self._invalid = 0
        return self._mtx