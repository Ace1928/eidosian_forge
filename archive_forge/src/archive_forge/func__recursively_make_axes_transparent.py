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
def _recursively_make_axes_transparent(exit_stack, ax):
    exit_stack.enter_context(ax.patch._cm_set(facecolor='none', edgecolor='none'))
    for child_ax in ax.child_axes:
        exit_stack.enter_context(child_ax.patch._cm_set(facecolor='none', edgecolor='none'))
    for child_childax in ax.child_axes:
        _recursively_make_axes_transparent(exit_stack, child_childax)