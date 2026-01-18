import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
def update_from_current_subplotpars(self):
    self._defaults = {spinbox: getattr(self._figure.subplotpars, name) for name, spinbox in self._spinboxes.items()}
    self._reset()