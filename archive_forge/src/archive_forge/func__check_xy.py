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
def _check_xy(self, renderer=None):
    """Check whether the annotation at *xy_pixel* should be drawn."""
    if renderer is None:
        renderer = self.figure._get_renderer()
    b = self.get_annotation_clip()
    if b or (b is None and self.xycoords == 'data'):
        xy_pixel = self._get_position_xy(renderer)
        return self.axes.contains_point(xy_pixel)
    return True