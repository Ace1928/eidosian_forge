import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_tickright(self):
    self._transform = Affine2D().scale(1.0, 1.0)
    self._snap_threshold = 1.0
    self._filled = False
    self._path = self._tickhoriz_path