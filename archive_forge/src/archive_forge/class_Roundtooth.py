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
class Roundtooth(Sawtooth):
    """A box with a rounded sawtooth outline."""

    def __call__(self, x0, y0, width, height, mutation_size):
        saw_vertices = self._get_sawtooth_vertices(x0, y0, width, height, mutation_size)
        saw_vertices = np.concatenate([saw_vertices, [saw_vertices[0]]])
        codes = [Path.MOVETO] + [Path.CURVE3, Path.CURVE3] * ((len(saw_vertices) - 1) // 2) + [Path.CLOSEPOLY]
        return Path(saw_vertices, codes)