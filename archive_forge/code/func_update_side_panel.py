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
def update_side_panel(self):
    current_fillings = [c.filling for c in self.manifold.cusp_info()]
    for n, coeffs in enumerate(current_fillings):
        for m in (0, 1):
            value = '%g' % coeffs[m]
            value = '0' if value == '-0' else value
            self.filling_vars[n][m].set(value)