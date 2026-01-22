import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
def _recalculate_axis_ticks(self, axis):
    b = self._bounding_box
    if b[axis][0] is None or b[axis][1] is None:
        self._axis_ticks[axis] = []
    else:
        self._axis_ticks[axis] = strided_range(b[axis][0], b[axis][1], self._stride[axis])