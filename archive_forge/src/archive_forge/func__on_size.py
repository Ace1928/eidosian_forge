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
def _on_size(self, event):
    """
        Called when wxEventSize is generated.

        In this application we attempt to resize to fit the window, so it
        is better to take the performance hit and redraw the whole window.
        """
    _log.debug('%s - _on_size()', type(self))
    sz = self.GetParent().GetSizer()
    if sz:
        si = sz.GetItem(self)
    if sz and si and (not si.Proportion) and (not si.Flag & wx.EXPAND):
        size = self.GetMinSize()
    else:
        size = self.GetClientSize()
        size.IncTo(self.GetMinSize())
    if getattr(self, '_width', None):
        if size == (self._width, self._height):
            return
    self._width, self._height = size
    self._isDrawn = False
    if self._width <= 1 or self._height <= 1:
        return
    self.bitmap = wx.Bitmap(self._width, self._height)
    dpival = self.figure.dpi
    winch = self._width / dpival
    hinch = self._height / dpival
    self.figure.set_size_inches(winch, hinch, forward=False)
    self.Refresh(eraseBackground=False)
    ResizeEvent('resize_event', self)._process()
    self.draw_idle()