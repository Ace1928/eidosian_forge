import os
import sys
import re
import time
from collections.abc import Mapping  # Python 3.5 or newer
from IPython.core.displayhook import DisplayHook
from tkinter.messagebox import askyesno
from .gui import *
from . import filedialog
from .exceptions import SnapPeaFatalError
from .app_menus import HelpMenu, EditMenu, WindowMenu, ListedWindow
from .app_menus import dirichlet_menus, horoball_menus, inside_view_menus, plink_menus
from .app_menus import add_menu, scut, open_html_docs
from .browser import Browser
from .horoviewer import HoroballViewer
from .infowindow import about_snappy, InfoWindow
from .polyviewer import PolyhedronViewer
from .raytracing.inside_viewer import InsideViewer
from .settings import Settings, SettingsDialog
from .phone_home import update_needed
from .SnapPy import SnapPea_interrupt, msg_stream
from .shell import SnapPyInteractiveShellEmbed
from .tkterminal import TkTerm, snappy_path
from plink import LinkEditor
from plink.smooth import Smoother
import site
import pydoc
class SnapPyBrowser(Browser, ListedWindow):

    def __init__(self, manifold, root=None, main_window=None):
        Browser.__init__(self, manifold, root=root, main_window=terminal)
        self.settings = terminal.settings
        self.menu_title = self.title()
        self.register_window(self)
        self.dirichlet_viewer.help_button.configure(command=self.dirichlet_help)

    def close(self, event=None):
        self.unregister_window(self)
        self.destroy()

    def apply_settings(self):
        if self.inside_view:
            self.inside_view.apply_settings(self.main_window.settings)

    def dirichlet_help(self):
        if not hasattr(self, 'polyhedron_help'):
            self.polyhedron_help = InfoWindow(self, 'Polyhedron Viewer Help', self.dirichlet_viewer.widget.help_text, 'polyhedron_help')
        else:
            self.polyhedron_help.deiconify()
            self.polyhedron_help.lift()
            self.polyhedron_help.focus_force()

    def horoball_help(self):
        if not hasattr(self, 'horoviewer_help'):
            self.horoviewer_help = InfoWindow(self, 'Horoball Viewer Help', self.horoball_viewer.widget.help_text, 'horoviewer_help')
        else:
            self.horoviewer_help.deiconify()
            self.horoviewer_help.lift()
            self.horoviewer_help.focus_force()