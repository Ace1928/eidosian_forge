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
def _extend_style_menu(self, style_menu):
    style_menu.add_radiobutton(label='Smooth edit', value='both', command=self.set_style, variable=self.style_var)