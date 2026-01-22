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
def _check_update(self):
    if self.state == 'start_state':
        return True
    elif self.state == 'dragging_state':
        x, y = (self.cursorx, self.canvas.winfo_height() - self.cursory)
        self.write_text('(%d, %d)' % (x, y))
    return False