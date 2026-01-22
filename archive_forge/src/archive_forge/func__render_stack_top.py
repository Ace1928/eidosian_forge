import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
def _render_stack_top(self, render_stack):
    top = render_stack[-1]
    if top == -1:
        return -1
    elif callable(top):
        dl = self._create_display_list(top)
        render_stack[-1] = (dl, top)
        return dl
    elif len(top) == 2:
        if pgl.GL_TRUE == pgl.glIsList(top[0]):
            return top[0]
        dl = self._create_display_list(top[1])
        render_stack[-1] = (dl, top[1])
        return dl