import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
def adjust_bounds(self, child_bounds):
    b = self._bounding_box
    c = child_bounds
    for i in range(3):
        if abs(c[i][0]) is S.Infinity or abs(c[i][1]) is S.Infinity:
            continue
        b[i][0] = c[i][0] if b[i][0] is None else min([b[i][0], c[i][0]])
        b[i][1] = c[i][1] if b[i][1] is None else max([b[i][1], c[i][1]])
        self._bounding_box = b
        self._recalculate_axis_ticks(i)