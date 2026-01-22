import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
def _on_key_down(self, event):
    """Capture key press."""
    KeyEvent('key_press_event', self, self._get_key(event), *self._mpl_coords(), guiEvent=event)._process()
    if self:
        event.Skip()