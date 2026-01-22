from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def _rendercursor(self):
    if self.ax.figure._get_renderer() is None:
        self.ax.figure.canvas.draw()
    text = self.text_disp.get_text()
    widthtext = text[:self.cursor_index]
    bb_text = self.text_disp.get_window_extent()
    self.text_disp.set_text(widthtext or ',')
    bb_widthtext = self.text_disp.get_window_extent()
    if bb_text.y0 == bb_text.y1:
        bb_text.y0 -= bb_widthtext.height / 2
        bb_text.y1 += bb_widthtext.height / 2
    elif not widthtext:
        bb_text.x1 = bb_text.x0
    else:
        bb_text.x1 = bb_text.x0 + bb_widthtext.width
    self.cursor.set(segments=[[(bb_text.x1, bb_text.y0), (bb_text.x1, bb_text.y1)]], visible=True)
    self.text_disp.set_text(text)
    self.ax.figure.canvas.draw()