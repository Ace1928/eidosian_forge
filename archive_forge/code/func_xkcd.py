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
def xkcd(scale: float=1, length: float=100, randomness: float=2) -> ExitStack:
    """
    Turn on `xkcd <https://xkcd.com/>`_ sketch-style drawing mode.

    This will only have an effect on things drawn after this function is called.

    For best results, install the `xkcd script <https://github.com/ipython/xkcd-font/>`_
    font; xkcd fonts are not packaged with Matplotlib.

    Parameters
    ----------
    scale : float, optional
        The amplitude of the wiggle perpendicular to the source line.
    length : float, optional
        The length of the wiggle along the line.
    randomness : float, optional
        The scale factor by which the length is shrunken or expanded.

    Notes
    -----
    This function works by a number of rcParams, so it will probably
    override others you have set before.

    If you want the effects of this function to be temporary, it can
    be used as a context manager, for example::

        with plt.xkcd():
            # This figure will be in XKCD-style
            fig1 = plt.figure()
            # ...

        # This figure will be in regular style
        fig2 = plt.figure()
    """
    if rcParams['text.usetex']:
        raise RuntimeError('xkcd mode is not compatible with text.usetex = True')
    stack = ExitStack()
    stack.callback(dict.update, rcParams, rcParams.copy())
    from matplotlib import patheffects
    rcParams.update({'font.family': ['xkcd', 'xkcd Script', 'Comic Neue', 'Comic Sans MS'], 'font.size': 14.0, 'path.sketch': (scale, length, randomness), 'path.effects': [patheffects.withStroke(linewidth=4, foreground='w')], 'axes.linewidth': 1.5, 'lines.linewidth': 2.0, 'figure.facecolor': 'white', 'grid.linewidth': 0.0, 'axes.grid': False, 'axes.unicode_minus': False, 'axes.edgecolor': 'black', 'xtick.major.size': 8, 'xtick.major.width': 3, 'ytick.major.size': 8, 'ytick.major.width': 3})
    return stack