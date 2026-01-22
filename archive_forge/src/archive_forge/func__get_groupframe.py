import uuid
import weakref
from contextlib import contextmanager
import logging
import math
import os.path
import pathlib
import sys
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
from tkinter.simpledialog import SimpleDialog
import numpy as np
from PIL import Image, ImageTk
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook, _c_internal_utils
from matplotlib.backend_bases import (
from matplotlib._pylab_helpers import Gcf
from . import _tkagg
def _get_groupframe(self, group):
    if group not in self._groups:
        if self._groups:
            self._add_separator()
        frame = tk.Frame(master=self, borderwidth=0)
        frame.pack(side=tk.LEFT, fill=tk.Y)
        frame._label_font = self._label_font
        self._groups[group] = frame
    return self._groups[group]