import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
class ToolGrid(ToolBase):
    """Tool to toggle the major grids of the figure."""
    description = 'Toggle major grids'
    default_keymap = property(lambda self: mpl.rcParams['keymap.grid'])

    def trigger(self, sender, event, data=None):
        sentinel = str(uuid.uuid4())
        with cbook._setattr_cm(event, key=sentinel), mpl.rc_context({'keymap.grid': sentinel}):
            mpl.backend_bases.key_press_handler(event, self.figure.canvas)