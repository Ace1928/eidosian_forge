import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
@_api.deprecated('3.7')
class BrokenBarHCollection(PolyCollection):
    """
    A collection of horizontal bars spanning *yrange* with a sequence of
    *xranges*.
    """

    def __init__(self, xranges, yrange, **kwargs):
        """
        Parameters
        ----------
        xranges : list of (float, float)
            The sequence of (left-edge-position, width) pairs for each bar.
        yrange : (float, float)
            The (lower-edge, height) common to all bars.
        **kwargs
            Forwarded to `.Collection`.
        """
        ymin, ywidth = yrange
        ymax = ymin + ywidth
        verts = [[(xmin, ymin), (xmin, ymax), (xmin + xwidth, ymax), (xmin + xwidth, ymin), (xmin, ymin)] for xmin, xwidth in xranges]
        super().__init__(verts, **kwargs)