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
def update_bbox_position_size(self, renderer):
    """
        Update the location and the size of the bbox.

        This method should be used when the position and size of the bbox needs
        to be updated before actually drawing the bbox.
        """
    if self._bbox_patch:
        posx = float(self.convert_xunits(self._x))
        posy = float(self.convert_yunits(self._y))
        posx, posy = self.get_transform().transform((posx, posy))
        x_box, y_box, w_box, h_box = _get_textbox(self, renderer)
        self._bbox_patch.set_bounds(0.0, 0.0, w_box, h_box)
        self._bbox_patch.set_transform(Affine2D().rotate_deg(self.get_rotation()).translate(posx + x_box, posy + y_box))
        fontsize_in_pixel = renderer.points_to_pixels(self.get_size())
        self._bbox_patch.set_mutation_scale(fontsize_in_pixel)