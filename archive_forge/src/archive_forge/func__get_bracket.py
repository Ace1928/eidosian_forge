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
def _get_bracket(self, x0, y0, x1, y1, width, length, angle):
    cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
    from matplotlib.bezier import get_normal_points
    x1, y1, x2, y2 = get_normal_points(x0, y0, cos_t, sin_t, width)
    dx, dy = (length * cos_t, length * sin_t)
    vertices_arrow = [(x1 + dx, y1 + dy), (x1, y1), (x2, y2), (x2 + dx, y2 + dy)]
    codes_arrow = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    if angle:
        trans = transforms.Affine2D().rotate_deg_around(x0, y0, angle)
        vertices_arrow = trans.transform(vertices_arrow)
    return (vertices_arrow, codes_arrow)