import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_plus_filled(self):
    self._transform = Affine2D()
    self._snap_threshold = 5.0
    self._joinstyle = self._user_joinstyle or JoinStyle.miter
    if not self._half_fill():
        self._path = self._plus_filled_path
    else:
        self._path = self._alt_path = self._plus_filled_path_t
        fs = self.get_fillstyle()
        self._transform.rotate_deg({'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[fs])
        self._alt_transform = self._transform.frozen().rotate_deg(180)