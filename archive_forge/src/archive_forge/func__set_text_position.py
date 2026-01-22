import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def _set_text_position(self, renderer):
    """Set text up so it is drawn in the right place."""
    bbox = self.get_window_extent(renderer)
    y = bbox.y0 + bbox.height / 2
    loc = self._text.get_horizontalalignment()
    if loc == 'center':
        x = bbox.x0 + bbox.width / 2
    elif loc == 'left':
        x = bbox.x0 + bbox.width * self.PAD
    else:
        x = bbox.x0 + bbox.width * (1 - self.PAD)
    self._text.set_position((x, y))