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
class SelectableText(ttk.Frame):
    """
    A Label and a disabled Entry widget which is disguised as a label
    but, unlike a Label, allows selecting the text.  On the Mac, the
    background color matches the background of a depth 2 LabelFrame
    by default.
    """

    def __init__(self, container, labeltext='', width=18, depth=2):
        ttk.Frame.__init__(self, container)
        self.var = Tk_.StringVar(container)
        style = SnapPyStyle()
        bg_color = style.groupBG if depth == 1 else style.subgroupBG
        self.label = label = ttk.Label(self, text=labeltext)
        self.value = value = Tk_.Entry(self, textvariable=self.var, state='readonly', borderwidth=0, readonlybackground=bg_color, highlightbackground=bg_color, highlightcolor=bg_color, highlightthickness=0, takefocus=False)
        if width:
            value.config(width=width)
        label.pack(side=Tk_.LEFT)
        value.pack(side=Tk_.LEFT, padx=2)

    def set(self, value):
        self.var.set(value)
        self.value.selection_clear()

    def get(self):
        return self.var.get()