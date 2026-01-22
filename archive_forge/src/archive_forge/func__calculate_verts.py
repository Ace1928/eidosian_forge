import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
def _calculate_verts(self):
    if self._calculating_verts.is_set():
        return
    self._calculating_verts.set()
    try:
        self._on_calculate_verts()
    finally:
        self._calculating_verts.clear()
    if callable(self.bounds_callback):
        self.bounds_callback()