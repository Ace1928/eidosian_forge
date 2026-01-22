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
def active_crossing_data(self):
    """
        Return the tuple of edges crossed by the in and out
        arrows of the active vertex.
        """
    assert self.ActiveVertex is not None
    active = self.ActiveVertex
    ignore = [active.in_arrow, active.out_arrow]
    return (self.crossed_arrows(active.in_arrow, ignore), self.crossed_arrows(active.out_arrow, ignore))