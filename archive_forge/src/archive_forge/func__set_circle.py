import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_circle(self, size=1.0):
    self._transform = Affine2D().scale(0.5 * size)
    self._snap_threshold = np.inf
    if not self._half_fill():
        self._path = Path.unit_circle()
    else:
        self._path = self._alt_path = Path.unit_circle_righthalf()
        fs = self.get_fillstyle()
        self._transform.rotate_deg({'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs])
        self._alt_transform = self._transform.frozen().rotate_deg(180.0)