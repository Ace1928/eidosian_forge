import sys
import os
from .gui import *
from .polyviewer import PolyhedronViewer
from .horoviewer import HoroballViewer
from .CyOpenGL import GetColor
from .app_menus import browser_menus
from . import app_menus
from .number import Number
from . import database
from .exceptions import SnapPeaFatalError
from plink import LinkViewer, LinkEditor
from plink.ipython_tools import IPythonTkRoot
from spherogram.links.orthogonal import OrthogonalLinkDiagram
def update_cusps(self):
    try:
        self.cusp_nbhd = self.manifold.cusp_neighborhood()
    except RuntimeError:
        self.cusp_nbhd = None
    self.horoball_viewer.new_scene(self.cusp_nbhd)
    self.after(100, self.horoball_viewer.cutoff_entry.selection_clear)