import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def _smooth_shift(self, key):
    try:
        ddx, ddy = vertex_shifts[key]
    except KeyError:
        return
    self.shifting = True
    dx, dy = self.shift_delta
    dx += ddx
    dy += ddy
    now = time.time()
    if now - self.shift_stamp < 0.1:
        self.shift_delta = (dx, dy)
    else:
        self.cursorx = x = self.ActiveVertex.x + dx
        self.cursory = y = self.ActiveVertex.y + dy
        self.move_active(x, y)
        self.shift_delta = (0, 0)
        self.shift_stamp = now