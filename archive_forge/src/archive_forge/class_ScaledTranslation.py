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
class ScaledTranslation(Affine2DBase):
    """
    A transformation that translates by *xt* and *yt*, after *xt* and *yt*
    have been transformed by *scale_trans*.
    """

    def __init__(self, xt, yt, scale_trans, **kwargs):
        super().__init__(**kwargs)
        self._t = (xt, yt)
        self._scale_trans = scale_trans
        self.set_children(scale_trans)
        self._mtx = None
        self._inverted = None
    __str__ = _make_str_method('_t')

    def get_matrix(self):
        if self._invalid:
            self._mtx = IdentityTransform._mtx.copy()
            self._mtx[:2, 2] = self._scale_trans.transform(self._t)
            self._invalid = 0
            self._inverted = None
        return self._mtx