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
def uninstall_repl_displayhook() -> None:
    """Disconnect from the display hook of the current shell."""
    global _REPL_DISPLAYHOOK
    if _REPL_DISPLAYHOOK is _ReplDisplayHook.IPYTHON:
        from IPython import get_ipython
        ip = get_ipython()
        ip.events.unregister('post_execute', _draw_all_if_interactive)
    _REPL_DISPLAYHOOK = _ReplDisplayHook.NONE