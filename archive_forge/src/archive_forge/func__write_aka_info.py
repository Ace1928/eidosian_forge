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
def _write_aka_info(self, mflds=None):
    aka_viewer = self.aka_viewer
    try:
        all_items = aka_viewer.get_children()
    except Tk_.TclError:
        return
    if all_items:
        aka_viewer.delete(*all_items)
    if mflds is None:
        aka_viewer.insert('', 'end', values=('Working ...', ''))
    else:
        strong = set(mflds['strong'])
        weak = set(mflds['weak']) - strong
        for mfld in strong:
            aka_viewer.insert('', 'end', values=(mfld, 'Yes'))
        for mfld in weak:
            aka_viewer.insert('', 'end', values=(mfld, 'No'))