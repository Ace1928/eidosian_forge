from __future__ import annotations
from contextlib import AbstractContextManager, ExitStack
from enum import Enum
import functools
import importlib
import inspect
import logging
import re
import sys
import threading
import time
from typing import cast, overload
from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import (  # Re-exported for typing.
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import _docstring
from matplotlib.backend_bases import (
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcsetup, rcParamsDefault, rcParamsOrig
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot  # type: ignore
from matplotlib.projections import PolarAxes  # type: ignore
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names
from matplotlib.cm import _colormaps
from matplotlib.cm import register_cmap  # type: ignore
from matplotlib.colors import _color_sequences
import numpy as np
from typing import TYPE_CHECKING, cast
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D, AxLine
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider, Widget
from .ticker import (
def ylim(*args, **kwargs) -> tuple[float, float]:
    """
    Get or set the y-limits of the current axes.

    Call signatures::

        bottom, top = ylim()  # return the current ylim
        ylim((bottom, top))   # set the ylim to bottom, top
        ylim(bottom, top)     # set the ylim to bottom, top

    If you do not specify args, you can alternatively pass *bottom* or
    *top* as kwargs, i.e.::

        ylim(top=3)  # adjust the top leaving bottom unchanged
        ylim(bottom=1)  # adjust the bottom leaving top unchanged

    Setting limits turns autoscaling off for the y-axis.

    Returns
    -------
    bottom, top
        A tuple of the new y-axis limits.

    Notes
    -----
    Calling this function with no arguments (e.g. ``ylim()``) is the pyplot
    equivalent of calling `~.Axes.get_ylim` on the current axes.
    Calling this function with arguments is the pyplot equivalent of calling
    `~.Axes.set_ylim` on the current axes. All arguments are passed though.
    """
    ax = gca()
    if not args and (not kwargs):
        return ax.get_ylim()
    ret = ax.set_ylim(*args, **kwargs)
    return ret