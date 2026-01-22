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
def _on_paint(self, event):
    """Called when wxPaintEvt is generated."""
    _log.debug('%s - _on_paint()', type(self))
    drawDC = wx.PaintDC(self)
    if not self._isDrawn:
        self.draw(drawDC=drawDC)
    else:
        self.gui_repaint(drawDC=drawDC)
    drawDC.Destroy()