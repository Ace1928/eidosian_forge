import numpy as np
import param
from ...element import HLine, HSpan, Tiles, VLine, VSpan
from ..mixins import GeomMixin
from .element import ElementPlot
class HVLinePlot(ShapePlot):
    apply_ranges = param.Boolean(default=False, doc='\n        Whether to include the annotation in axis range calculations.')
    _shape_type = 'line'
    _supports_geo = False

    def get_data(self, element, ranges, style, **kwargs):
        if isinstance(element, HLine) and self.invert_axes or (isinstance(element, VLine) and (not self.invert_axes)):
            x = element.data
            visible = x is not None
            return [dict(x0=x, x1=x, y0=0, y1=1, xref='x', yref='paper', visible=visible)]
        else:
            y = element.data
            visible = y is not None
            return [dict(x0=0.0, x1=1.0, y0=y, y1=y, xref='paper', yref='y', visible=visible)]