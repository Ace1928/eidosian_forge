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
def _in_patch(self, patch):
    """
            Return a predicate function testing whether a point *xy* is
            contained in *patch*.
            """
    return lambda xy: patch.contains(SimpleNamespace(x=xy[0], y=xy[1]))[0]