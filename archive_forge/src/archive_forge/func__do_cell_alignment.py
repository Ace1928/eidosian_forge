import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def _do_cell_alignment(self):
    """
        Calculate row heights and column widths; position cells accordingly.
        """
    widths = {}
    heights = {}
    for (row, col), cell in self._cells.items():
        height = heights.setdefault(row, 0.0)
        heights[row] = max(height, cell.get_height())
        width = widths.setdefault(col, 0.0)
        widths[col] = max(width, cell.get_width())
    xpos = 0
    lefts = {}
    for col in sorted(widths):
        lefts[col] = xpos
        xpos += widths[col]
    ypos = 0
    bottoms = {}
    for row in sorted(heights, reverse=True):
        bottoms[row] = ypos
        ypos += heights[row]
    for (row, col), cell in self._cells.items():
        cell.set_x(lefts[col])
        cell.set_y(bottoms[row])