from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
@staticmethod
def _norm_per_subplot_kw(per_subplot_kw):
    expanded = {}
    for k, v in per_subplot_kw.items():
        if isinstance(k, tuple):
            for sub_key in k:
                if sub_key in expanded:
                    raise ValueError(f'The key {sub_key!r} appears multiple times.')
                expanded[sub_key] = v
        else:
            if k in expanded:
                raise ValueError(f'The key {k!r} appears multiple times.')
            expanded[k] = v
    return expanded