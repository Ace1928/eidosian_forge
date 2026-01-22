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
def _print_image(self, filetype, filename):
    bitmap = wx.Bitmap(math.ceil(self.figure.bbox.width), math.ceil(self.figure.bbox.height))
    self.figure.draw(RendererWx(bitmap, self.figure.dpi))
    saved_obj = bitmap.ConvertToImage() if cbook.is_writable_file_like(filename) else bitmap
    if not saved_obj.SaveFile(filename, filetype):
        raise RuntimeError(f'Could not save figure to {filename}')
    if self._isDrawn:
        self.draw()
    if self:
        self.Refresh()