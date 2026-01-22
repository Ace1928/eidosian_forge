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
class ToolQuit(ToolBase):
    """Tool to call the figure manager destroy method."""
    description = 'Quit the figure'
    default_keymap = property(lambda self: mpl.rcParams['keymap.quit'])

    def trigger(self, sender, event, data=None):
        Gcf.destroy_fig(self.figure)