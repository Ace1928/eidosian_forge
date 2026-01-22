import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_pixel(self):
    self._path = Path.unit_rectangle()
    self._transform = Affine2D().translate(-0.49999, -0.49999)
    self._snap_threshold = None