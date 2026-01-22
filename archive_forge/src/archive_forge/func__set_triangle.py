import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_triangle(self, rot, skip):
    self._transform = Affine2D().scale(0.5).rotate_deg(rot)
    self._snap_threshold = 5.0
    if not self._half_fill():
        self._path = self._triangle_path
    else:
        mpaths = [self._triangle_path_u, self._triangle_path_l, self._triangle_path_d, self._triangle_path_r]
        fs = self.get_fillstyle()
        if fs == 'top':
            self._path = mpaths[(0 + skip) % 4]
            self._alt_path = mpaths[(2 + skip) % 4]
        elif fs == 'bottom':
            self._path = mpaths[(2 + skip) % 4]
            self._alt_path = mpaths[(0 + skip) % 4]
        elif fs == 'left':
            self._path = mpaths[(1 + skip) % 4]
            self._alt_path = mpaths[(3 + skip) % 4]
        else:
            self._path = mpaths[(3 + skip) % 4]
            self._alt_path = mpaths[(1 + skip) % 4]
        self._alt_transform = self._transform
    self._joinstyle = self._user_joinstyle or JoinStyle.miter