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
def _auto_legend_data(self):
    """
        Return display coordinates for hit testing for "best" positioning.

        Returns
        -------
        bboxes
            List of bounding boxes of all patches.
        lines
            List of `.Path` corresponding to each line.
        offsets
            List of (x, y) offsets of all collection.
        """
    assert self.isaxes
    bboxes = []
    lines = []
    offsets = []
    for artist in self.parent._children:
        if isinstance(artist, Line2D):
            lines.append(artist.get_transform().transform_path(artist.get_path()))
        elif isinstance(artist, Rectangle):
            bboxes.append(artist.get_bbox().transformed(artist.get_data_transform()))
        elif isinstance(artist, Patch):
            lines.append(artist.get_transform().transform_path(artist.get_path()))
        elif isinstance(artist, Collection):
            transform, transOffset, hoffsets, _ = artist._prepare_points()
            if len(hoffsets):
                for offset in transOffset.transform(hoffsets):
                    offsets.append(offset)
    return (bboxes, lines, offsets)