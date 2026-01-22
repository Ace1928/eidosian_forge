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
class Fancy(_Base):
    """A fancy arrow. Only works with a quadratic Bézier curve."""

    def __init__(self, head_length=0.4, head_width=0.4, tail_width=0.4):
        """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.4
                Width of the arrow head.

            tail_width : float, default: 0.4
                Width of the arrow tail.
            """
        self.head_length, self.head_width, self.tail_width = (head_length, head_width, tail_width)
        super().__init__()

    def transmute(self, path, mutation_size, linewidth):
        x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)
        head_length = self.head_length * mutation_size
        arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
        in_f = inside_circle(x2, y2, head_length)
        try:
            path_out, path_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
        except NonIntersectingPathException:
            x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
            x1n, y1n = (0.5 * (x0 + x2), 0.5 * (y0 + y2))
            arrow_path = [(x0, y0), (x1n, y1n), (x2, y2)]
            path_head = arrow_path
        else:
            path_head = path_in
        in_f = inside_circle(x2, y2, head_length * 0.8)
        path_out, path_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
        path_tail = path_out
        head_width = self.head_width * mutation_size
        head_l, head_r = make_wedged_bezier2(path_head, head_width / 2.0, wm=0.6)
        tail_width = self.tail_width * mutation_size
        tail_left, tail_right = make_wedged_bezier2(path_tail, tail_width * 0.5, w1=1.0, wm=0.6, w2=0.3)
        in_f = inside_circle(x0, y0, tail_width * 0.3)
        path_in, path_out = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
        tail_start = path_in[-1]
        head_right, head_left = (head_r, head_l)
        patch_path = [(Path.MOVETO, tail_start), (Path.LINETO, tail_right[0]), (Path.CURVE3, tail_right[1]), (Path.CURVE3, tail_right[2]), (Path.LINETO, head_right[0]), (Path.CURVE3, head_right[1]), (Path.CURVE3, head_right[2]), (Path.CURVE3, head_left[1]), (Path.CURVE3, head_left[0]), (Path.LINETO, tail_left[2]), (Path.CURVE3, tail_left[1]), (Path.CURVE3, tail_left[0]), (Path.LINETO, tail_start), (Path.CLOSEPOLY, tail_start)]
        path = Path([p for c, p in patch_path], [c for c, p in patch_path])
        return (path, True)