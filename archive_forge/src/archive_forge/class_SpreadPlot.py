import numpy as np
import param
from ...element import Tiles
from ...operation import interpolate_curve
from ..mixins import AreaMixin, BarsMixin
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class SpreadPlot(ChartPlot):
    padding = param.ClassSelector(default=(0, 0.1), class_=(int, float, tuple))
    style_opts = ['visible', 'color', 'dash', 'line_width']
    _style_key = 'line'

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'scatter', 'mode': 'lines'}

    def get_data(self, element, ranges, style, **kwargs):
        x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
        xs = element.dimension_values(0)
        mean = element.dimension_values(1)
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        lower = mean - neg_error
        upper = mean + pos_error
        return [{x: xs, y: lower, 'fill': None}, {x: xs, y: upper, 'fill': 'tonext' + y}]