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
def _remove_vertex(self, i):
    """Remove vertex with index i."""
    if len(self._xys) > 2 and self._selection_completed and (i in (0, len(self._xys) - 1)):
        self._xys.pop(0)
        self._xys.pop(-1)
        self._xys.append(self._xys[0])
    else:
        self._xys.pop(i)
    if len(self._xys) <= 2:
        self._selection_completed = False
        self._remove_box()