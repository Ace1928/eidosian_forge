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
class SnapPySettings(Settings, ListedWindow):

    def __init__(self, terminal):
        self.terminal = terminal
        Settings.__init__(self, terminal.text)
        self.apply_settings()

    def apply_settings(self):
        self.terminal.set_font(self['font'])
        changed = self.changed()
        IP = self.terminal.IP
        self.terminal.quiet = True
        if 'autocall' in changed:
            if self.setting_dict['autocall']:
                IP.magics_manager.magics['line']['autocall'](2)
            else:
                IP.magics_manager.magics['line']['autocall'](0)
        if 'automagic' in changed:
            if self.setting_dict['automagic']:
                IP.magics_manager.magics['line']['automagic']('on')
            else:
                IP.magics_manager.magics['line']['automagic']('off')
        self.terminal.quiet = False
        for window in self.window_list:
            window.apply_settings()