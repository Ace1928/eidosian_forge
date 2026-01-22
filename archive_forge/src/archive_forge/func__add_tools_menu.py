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
def _add_tools_menu(self):
    self.lock_var = Tk_.BooleanVar(self.window)
    self.lock_var.set(False)
    self.tools_menu = tools_menu = Tk_.Menu(self.menubar, tearoff=0)
    tools_menu.add_command(label='Make alternating', command=self.make_alternating)
    tools_menu.add_command(label='Reflect', command=self.reflect)
    tools_menu.add_checkbutton(label='Preserve diagram', var=self.lock_var)
    tools_menu.add_command(label='Clear', command=self.clear)
    if self.callback:
        tools_menu.add_command(label=self.cb_menu, command=self._do_callback)
    self.menubar.add_cascade(label='Tools', menu=tools_menu)