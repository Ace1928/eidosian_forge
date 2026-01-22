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
def _on_motion(self, event):
    """Start measuring on an axis."""
    event.Skip()
    MouseEvent('motion_notify_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(event), guiEvent=event)._process()