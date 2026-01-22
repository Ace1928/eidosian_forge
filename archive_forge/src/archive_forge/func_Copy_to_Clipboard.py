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
def Copy_to_Clipboard(self, event=None):
    """Copy bitmap of canvas to system clipboard."""
    bmp_obj = wx.BitmapDataObject()
    bmp_obj.SetBitmap(self.bitmap)
    if not wx.TheClipboard.IsOpened():
        open_success = wx.TheClipboard.Open()
        if open_success:
            wx.TheClipboard.SetData(bmp_obj)
            wx.TheClipboard.Flush()
            wx.TheClipboard.Close()