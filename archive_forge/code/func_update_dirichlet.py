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
def update_dirichlet(self):
    try:
        self.dirichlet = self.manifold.dirichlet_domain().face_list()
    except RuntimeError:
        self.dirichlet = []
    self.update_modeline()
    self.dirichlet_viewer.new_polyhedron(self.dirichlet)
    if self.inside_view:
        self.inside_view.show_failed_dirichlet(show=len(self.dirichlet) == 0)