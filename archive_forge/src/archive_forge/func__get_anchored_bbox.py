import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
def _get_anchored_bbox(loc, bbox, parentbbox, borderpad):
    """
    Return the (x, y) position of the *bbox* anchored at the *parentbbox* with
    the *loc* code with the *borderpad*.
    """
    c = [None, 'NE', 'NW', 'SW', 'SE', 'E', 'W', 'E', 'S', 'N', 'C'][loc]
    container = parentbbox.padded(-borderpad)
    return bbox.anchored(c, container=container).p0