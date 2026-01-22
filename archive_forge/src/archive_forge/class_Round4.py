import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
@_register_style(_style_list)
class Round4:
    """A box with rounded edges."""

    def __init__(self, pad=0.3, rounding_size=None):
        """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            rounding_size : float, default: *pad*/2
                Rounding of edges.
            """
        self.pad = pad
        self.rounding_size = rounding_size

    def __call__(self, x0, y0, width, height, mutation_size):
        pad = mutation_size * self.pad
        if self.rounding_size:
            dr = mutation_size * self.rounding_size
        else:
            dr = pad / 2.0
        width = width + 2 * pad - 2 * dr
        height = height + 2 * pad - 2 * dr
        x0, y0 = (x0 - pad + dr, y0 - pad + dr)
        x1, y1 = (x0 + width, y0 + height)
        cp = [(x0, y0), (x0 + dr, y0 - dr), (x1 - dr, y0 - dr), (x1, y0), (x1 + dr, y0 + dr), (x1 + dr, y1 - dr), (x1, y1), (x1 - dr, y1 + dr), (x0 + dr, y1 + dr), (x0, y1), (x0 - dr, y1 - dr), (x0 - dr, y0 + dr), (x0, y0), (x0, y0)]
        com = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        return Path(cp, com)