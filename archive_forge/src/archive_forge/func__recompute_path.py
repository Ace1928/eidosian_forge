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
def _recompute_path(self):
    arc = Path.arc(0, 360)
    a, b, w = (self.a, self.b, self.width)
    v1 = self._transform_verts(arc.vertices, a, b)
    v2 = self._transform_verts(arc.vertices[::-1], a - w, b - w)
    v = np.vstack([v1, v2, v1[0, :], (0, 0)])
    c = np.hstack([arc.codes, Path.MOVETO, arc.codes[1:], Path.MOVETO, Path.CLOSEPOLY])
    self._path = Path(v, c)