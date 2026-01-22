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
def _cancel_action(self):
    self._button_pressed = None
    self._xypress = []
    self.figure.canvas.mpl_disconnect(self._id_drag)
    self.toolmanager.messagelock.release(self)
    self.figure.canvas.draw_idle()