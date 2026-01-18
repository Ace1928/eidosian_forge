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
def update_aka(self):
    self._write_aka_info()
    M = self.manifold.copy()

    def format_name(N):
        if all(N.cusp_info('is_complete')):
            return N.name()
        return repr(N)
    try:
        mflds = {'weak': [format_name(N) for N in M.identify()], 'strong': [format_name(N) for N in M.identify(True)]}
    except ValueError:
        mflds = {'weak': [], 'strong': []}
    self._write_aka_info(mflds)