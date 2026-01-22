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
@backend_tools._register_tool_class(_FigureCanvasWxBase)
class ConfigureSubplotsWx(backend_tools.ConfigureSubplotsBase):

    def trigger(self, *args):
        NavigationToolbar2Wx.configure_subplots(self)