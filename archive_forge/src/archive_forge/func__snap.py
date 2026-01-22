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
@staticmethod
def _snap(values, snap_values):
    """Snap values to a given array values (snap_values)."""
    eps = np.min(np.abs(np.diff(snap_values))) * 1e-12
    return tuple((snap_values[np.abs(snap_values - v + np.sign(v) * eps).argmin()] for v in values))