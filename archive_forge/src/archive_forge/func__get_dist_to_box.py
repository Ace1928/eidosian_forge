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
def _get_dist_to_box(self, rotation, x0, y0, figure_box):
    """
        Return the distance from the given points to the boundaries of a
        rotated box, in pixels.
        """
    if rotation > 270:
        quad = rotation - 270
        h1 = y0 / math.cos(math.radians(quad))
        h2 = (figure_box.x1 - x0) / math.cos(math.radians(90 - quad))
    elif rotation > 180:
        quad = rotation - 180
        h1 = x0 / math.cos(math.radians(quad))
        h2 = y0 / math.cos(math.radians(90 - quad))
    elif rotation > 90:
        quad = rotation - 90
        h1 = (figure_box.y1 - y0) / math.cos(math.radians(quad))
        h2 = x0 / math.cos(math.radians(90 - quad))
    else:
        h1 = (figure_box.x1 - x0) / math.cos(math.radians(rotation))
        h2 = (figure_box.y1 - y0) / math.cos(math.radians(90 - rotation))
    return min(h1, h2)