import itertools
import logging
import numbers
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
from matplotlib.collections import (
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler
def _findoffset(self, width, height, xdescent, ydescent, renderer):
    """Helper function to locate the legend."""
    if self._loc == 0:
        x, y = self._find_best_position(width, height, renderer)
    elif self._loc in Legend.codes.values():
        bbox = Bbox.from_bounds(0, 0, width, height)
        x, y = self._get_anchored_bbox(self._loc, bbox, self.get_bbox_to_anchor(), renderer)
    else:
        fx, fy = self._loc
        bbox = self.get_bbox_to_anchor()
        x, y = (bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy)
    return (x + xdescent, y + ydescent)