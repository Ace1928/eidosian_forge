import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
@xycoords.setter
def xycoords(self, xycoords):

    def is_offset(s):
        return isinstance(s, str) and s.startswith('offset')
    if isinstance(xycoords, tuple) and any(map(is_offset, xycoords)) or is_offset(xycoords):
        raise ValueError('xycoords cannot be an offset coordinate')
    self._xycoords = xycoords