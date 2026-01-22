from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def _max_in_bounds(self, max):
    """Ensure the new max value is between valmax and self.val[0]."""
    if max >= self.valmax:
        if not self.closedmax:
            return self.val[1]
        max = self.valmax
    if max <= self.val[0]:
        max = self.val[0]
    return self._stepped_value(max)