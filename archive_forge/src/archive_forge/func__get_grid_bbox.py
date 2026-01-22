import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def _get_grid_bbox(self, renderer):
    """
        Get a bbox, in axes coordinates for the cells.

        Only include those in the range (0, 0) to (maxRow, maxCol).
        """
    boxes = [cell.get_window_extent(renderer) for (row, col), cell in self._cells.items() if row >= 0 and col >= 0]
    bbox = Bbox.union(boxes)
    return bbox.transformed(self.get_transform().inverted())