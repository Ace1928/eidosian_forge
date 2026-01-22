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
def _update_path(self):
    stretched = self._theta_stretch()
    if any((a != b for a, b in zip(stretched, (self._theta1, self._theta2, self._stretched_width, self._stretched_height)))):
        self._theta1, self._theta2, self._stretched_width, self._stretched_height = stretched
        self._path = Path.arc(self._theta1, self._theta2)