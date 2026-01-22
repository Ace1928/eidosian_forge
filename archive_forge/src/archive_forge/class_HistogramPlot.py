import numpy as np
import param
from ...element import Tiles
from ...operation import interpolate_curve
from ..mixins import AreaMixin, BarsMixin
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class HistogramPlot(ElementPlot):
    style_opts = ['visible', 'color', 'line_color', 'line_width', 'opacity', 'selectedpoints']
    _style_key = 'marker'
    selection_display = PlotlyOverlaySelectionDisplay()

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'bar'}

    def get_data(self, element, ranges, style, **kwargs):
        xdim = element.kdims[0]
        ydim = element.vdims[0]
        values = np.asarray(element.interface.coords(element, ydim))
        edges = np.asarray(element.interface.coords(element, xdim))
        if len(edges) < 2:
            binwidth = 0
        else:
            binwidth = edges[1] - edges[0]
        if self.invert_axes:
            ys = edges
            xs = values
            orientation = 'h'
        else:
            xs = edges
            ys = values
            orientation = 'v'
        return [{'x': xs, 'y': ys, 'width': binwidth, 'orientation': orientation}]

    def init_layout(self, key, element, ranges, **kwargs):
        layout = super().init_layout(key, element, ranges)
        layout['barmode'] = 'overlay'
        return layout