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
def update_symmetry(self):
    """update_symmetry"""
    try:
        self.symmetry_group = self.manifold.symmetry_group()
    except (ValueError, SnapPeaFatalError):
        self.symmetry_group = str('unknown')
    self.symmetry.set(str(self.symmetry_group))