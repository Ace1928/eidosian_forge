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
class Coverer(SimpleDialog):

    def __init__(self, parent, manifold):
        self.manifold = manifold.copy()
        self.num = 0
        self.result = []
        style = SnapPyStyle()
        self.root = root = Tk_.Toplevel(parent, class_='SnapPy', bg=style.windowBG)
        title = 'Cover'
        root.title(title)
        root.iconname(title)
        root.bind('<Return>', self.handle_return)
        top_frame = ttk.Frame(root)
        top_frame.grid_rowconfigure(2, weight=1)
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)
        msg_font = Font(family=style.font_info['family'], weight='bold', size=int(style.font_info['size'] * 1.2))
        msg = ttk.Label(top_frame, font=msg_font, text='Choose covering spaces to browse:')
        msg.grid(row=0, column=0, columnspan=3, pady=10)
        degree_frame = ttk.Frame(top_frame)
        degree_frame.grid_columnconfigure(1, weight=1)
        self.degree_var = degree_var = Tk_.StringVar()
        degree_var.trace('w', self.show_covers)
        ttk.Label(degree_frame, text='Degree: ').grid(row=0, column=0, sticky=Tk_.E)
        self.degree_option = degree_option = ttk.OptionMenu(degree_frame, degree_var, None, *range(2, 9))
        degree_option.grid(row=0, column=1)
        self.cyclic_var = cyclic_var = Tk_.BooleanVar()
        cyclic_var.trace('w', self.show_covers)
        cyclic_or_not = ttk.Checkbutton(degree_frame, variable=cyclic_var, text='cyclic covers only')
        cyclic_or_not.grid(row=0, column=2, padx=6, sticky=Tk_.W)
        self.action = action = ttk.Button(degree_frame, text='Recompute', command=self.show_covers)
        action.grid(row=0, column=3, padx=8, sticky=Tk_.W)
        degree_frame.grid(row=1, column=0, pady=2, padx=6, sticky=Tk_.EW)
        self.covers = covers = ttk.Treeview(top_frame, selectmode='extended', columns=['index', 'cover_type', 'num_cusps', 'homology'], show='headings')
        covers.heading('index', text='')
        covers.column('index', stretch=False, width=40, minwidth=40)
        covers.heading('cover_type', text='Type')
        covers.column('cover_type', stretch=False, width=100)
        covers.heading('num_cusps', text='# Cusps')
        covers.column('num_cusps', stretch=False, width=100, anchor=Tk_.CENTER)
        covers.heading('homology', text='Homology')
        covers.column('homology', stretch=True, width=300)
        covers.bind('<Double-Button-1>', self.choose)
        self.covers.grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky=Tk_.NSEW)
        top_frame.pack(fill=Tk_.BOTH, expand=1)
        button_frame = ttk.Frame(self.root)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        self.browse = ttk.Button(button_frame, text='Browse', command=self.choose, default='active')
        self.browse.grid(row=0, column=0, sticky=Tk_.E, padx=6)
        button = ttk.Button(button_frame, text='Cancel', command=self.cancel)
        button.grid(row=0, column=1, sticky=Tk_.W, padx=6)
        button_frame.pack(pady=6, fill=Tk_.BOTH, expand=1)
        self.root.protocol('WM_DELETE_WINDOW', self.cancel)
        try:
            _place_window(self.root, parent)
        except AttributeError:
            self._set_transient(container)
        degree_var.set('2')
        cyclic_var.set(True)
        self.show_covers()

    def clear_list(self, *args):
        self.covers.delete(*self.covers.get_children())
        self.browse.config(default='normal')
        self.action.config(default='active')
        self.state = 'not ready'

    def show_covers(self, *args):
        self.state = 'ready'
        self.browse.config(default='active')
        self.action.config(default='normal')
        self.covers.delete(*self.covers.get_children())
        degree = int(self.degree_var.get())
        if self.cyclic_var.get():
            self.cover_list = self.manifold.covers(degree, cover_type='cyclic')
        else:
            self.cover_list = self.manifold.covers(degree)
        for n, N in enumerate(self.cover_list):
            cusps = repr(N.num_cusps())
            homology = repr(N.homology())
            name = N.name()
            cover_type = N.cover_info()['type']
            self.covers.insert('', 'end', values=(n, cover_type, cusps, homology))

    def handle_return(self, event):
        if self.state == 'ready':
            self.choose()
        else:
            self.show_covers()

    def choose(self, event=None):
        self.result = [self.cover_list[self.covers.index(x)] for x in self.covers.selection()]
        self.root.destroy()

    def cancel(self):
        self.result = []
        self.root.destroy()

    def go(self):
        self.root.grab_set()
        self.root.wait_window()