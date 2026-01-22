import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def _get_markerfacecolor(self, alt=False):
    if self._marker.get_fillstyle() == 'none':
        return 'none'
    fc = self._markerfacecoloralt if alt else self._markerfacecolor
    if cbook._str_lower_equal(fc, 'auto'):
        return self._color
    else:
        return fc