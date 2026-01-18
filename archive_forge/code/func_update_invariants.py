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
def update_invariants(self):
    manifold = self.manifold
    if not self.recompute_invariants:
        return
    self.orientability.set('Yes' if manifold.is_orientable() else 'No')
    try:
        self.volume.set(repr(manifold.volume()))
    except ValueError:
        self.volume.set('')
    try:
        self.cs.set(repr(manifold.chern_simons()))
    except ValueError:
        self.cs.set('')
    try:
        self.homology.set(repr(manifold.homology()))
    except ValueError:
        self.homology.set('')
    self.compute_pi_one()
    self.update_length_spectrum()
    self.update_dirichlet()
    self.update_aka()
    self.recompute_invariants = False