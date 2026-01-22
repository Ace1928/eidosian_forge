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
def _update_loc(self, loc_in_canvas):
    bbox = self.legend.get_bbox_to_anchor()
    if bbox.width == 0 or bbox.height == 0:
        self.legend.set_bbox_to_anchor(None)
        bbox = self.legend.get_bbox_to_anchor()
    _bbox_transform = BboxTransformFrom(bbox)
    self.legend._loc = tuple(_bbox_transform.transform(loc_in_canvas))