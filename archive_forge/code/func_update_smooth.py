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
def update_smooth(self):
    self.smoother.clear()
    mode = self.style_var.get()
    if mode == 'smooth':
        self.smoother.set_polylines(self.polylines())
    elif mode == 'both':
        self.smoother.set_polylines(self.polylines(), thickness=2)