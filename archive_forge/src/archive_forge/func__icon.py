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
@staticmethod
def _icon(name):
    """
        Construct a `wx.Bitmap` suitable for use as icon from an image file
        *name*, including the extension and relative to Matplotlib's "images"
        data directory.
        """
    pilimg = PIL.Image.open(cbook._get_data_path('images', name))
    image = np.array(pilimg.convert('RGBA'))
    try:
        dark = wx.SystemSettings.GetAppearance().IsDark()
    except AttributeError:
        bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
        fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        bg_lum = (0.299 * bg.red + 0.587 * bg.green + 0.114 * bg.blue) / 255
        fg_lum = (0.299 * fg.red + 0.587 * fg.green + 0.114 * fg.blue) / 255
        dark = fg_lum - bg_lum > 0.2
    if dark:
        fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        black_mask = (image[..., :3] == 0).all(axis=-1)
        image[black_mask, :3] = (fg.Red(), fg.Green(), fg.Blue())
    return wx.Bitmap.FromBufferRGBA(image.shape[1], image.shape[0], image.tobytes())