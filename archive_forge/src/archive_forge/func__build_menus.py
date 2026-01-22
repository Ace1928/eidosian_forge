import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def _build_menus(self):
    self.menubar = menubar = Tk_.Menu(self.window)
    self._add_file_menu()
    self._add_info_menu()
    self._add_tools_menu()
    self._add_style_menu()
    self.window.config(menu=menubar)
    help_menu = Tk_.Menu(menubar, tearoff=0)
    help_menu.add_command(label='About PLink...', command=self.about)
    help_menu.add_command(label='Instructions ...', command=self.howto)
    menubar.add_cascade(label='Help', menu=help_menu)