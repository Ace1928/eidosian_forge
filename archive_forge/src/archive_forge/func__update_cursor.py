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
def _update_cursor(self, event):
    """
        Update the cursor after a mouse move event or a tool (de)activation.
        """
    if self.mode and event.inaxes and event.inaxes.get_navigate():
        if self.mode == _Mode.ZOOM and self._last_cursor != tools.Cursors.SELECT_REGION:
            self.canvas.set_cursor(tools.Cursors.SELECT_REGION)
            self._last_cursor = tools.Cursors.SELECT_REGION
        elif self.mode == _Mode.PAN and self._last_cursor != tools.Cursors.MOVE:
            self.canvas.set_cursor(tools.Cursors.MOVE)
            self._last_cursor = tools.Cursors.MOVE
    elif self._last_cursor != tools.Cursors.POINTER:
        self.canvas.set_cursor(tools.Cursors.POINTER)
        self._last_cursor = tools.Cursors.POINTER