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
def update_current_tab(self, event=None):
    self.update_modeline_and_side_panel()
    tab_name = self.notebook.tab(self.notebook.select(), 'text')
    if tab_name == 'Invariants':
        self.update_menus(self.menubar)
        self.update_invariants()
    elif tab_name == 'Cusp Nbhds':
        self.horoball_viewer.update_menus(self.menubar)
        self.horoball_viewer.view_menu.focus_set()
        if self.horoball_viewer.empty:
            self.update_cusps()
        else:
            self.horoball_viewer.redraw()
    elif tab_name == 'Dirichlet':
        self.dirichlet_viewer.update_menus(self.menubar)
    elif tab_name == 'Link':
        self.update_menus(self.menubar)
        self.link_tab.draw()
    elif tab_name == 'Symmetry':
        self.update_menus(self.menubar)
        self.update_symmetry()
    else:
        self.update_menus(self.menubar)
    self.update_idletasks()