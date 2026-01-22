import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def _append_path(ctx, path, transform, clip=None):
    for points, code in path.iter_segments(transform, remove_nans=True, clip=clip):
        if code == Path.MOVETO:
            ctx.move_to(*points)
        elif code == Path.CLOSEPOLY:
            ctx.close_path()
        elif code == Path.LINETO:
            ctx.line_to(*points)
        elif code == Path.CURVE3:
            cur = np.asarray(ctx.get_current_point())
            a = points[:2]
            b = points[-2:]
            ctx.curve_to(*cur / 3 + a * 2 / 3, *a * 2 / 3 + b / 3, *b)
        elif code == Path.CURVE4:
            ctx.curve_to(*points)