from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
def _fill_intervals(self, intervals):
    self.intervals = [PlotInterval(i) for i in self.intervals]
    v_used = []
    for i in range(len(intervals)):
        self.intervals[i].fill_from(intervals[i])
        if self.intervals[i].v is not None:
            v_used.append(self.intervals[i].v)
    for i in range(len(self.intervals)):
        if self.intervals[i].v is None:
            u = [v for v in self.i_vars if v not in v_used]
            if len(u) == 0:
                raise ValueError('length should not be equal to 0')
            self.intervals[i].v = u[0]
            v_used.append(u[0])