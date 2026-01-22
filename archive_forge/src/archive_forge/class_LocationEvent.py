from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
class LocationEvent(Event):
    """
    An event that has a screen location.

    A LocationEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    x, y : int or None
        Event location in pixels from bottom left of canvas.
    inaxes : `~matplotlib.axes.Axes` or None
        The `~.axes.Axes` instance over which the mouse is, if any.
    xdata, ydata : float or None
        Data coordinates of the mouse within *inaxes*, or *None* if the mouse
        is not over an Axes.
    modifiers : frozenset
        The keyboard modifiers currently being pressed (except for KeyEvent).
    """
    _lastevent = None
    lastevent = _api.deprecated('3.8')(_api.classproperty(lambda cls: cls._lastevent))
    _last_axes_ref = None

    def __init__(self, name, canvas, x, y, guiEvent=None, *, modifiers=None):
        super().__init__(name, canvas, guiEvent=guiEvent)
        self.x = int(x) if x is not None else x
        self.y = int(y) if y is not None else y
        self.inaxes = None
        self.xdata = None
        self.ydata = None
        self.modifiers = frozenset(modifiers if modifiers is not None else [])
        if x is None or y is None:
            return
        self._set_inaxes(self.canvas.inaxes((x, y)) if self.canvas.mouse_grabber is None else self.canvas.mouse_grabber, (x, y))

    def _set_inaxes(self, inaxes, xy=None):
        self.inaxes = inaxes
        if inaxes is not None:
            try:
                self.xdata, self.ydata = inaxes.transData.inverted().transform(xy if xy is not None else (self.x, self.y))
            except ValueError:
                pass