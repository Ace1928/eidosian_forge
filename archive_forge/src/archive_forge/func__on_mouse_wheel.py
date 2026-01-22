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
def _on_mouse_wheel(self, event):
    """Translate mouse wheel events into matplotlib events"""
    x, y = self._mpl_coords(event)
    step = event.LinesPerAction * event.WheelRotation / event.WheelDelta
    event.Skip()
    if wx.Platform == '__WXMAC__':
        if not hasattr(self, '_skipwheelevent'):
            self._skipwheelevent = True
        elif self._skipwheelevent:
            self._skipwheelevent = False
            return
        else:
            self._skipwheelevent = True
    MouseEvent('scroll_event', self, x, y, step=step, modifiers=self._mpl_modifiers(event), guiEvent=event)._process()