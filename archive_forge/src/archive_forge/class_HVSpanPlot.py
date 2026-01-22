import numpy as np
import param
from ...element import HLine, HSpan, Tiles, VLine, VSpan
from ..mixins import GeomMixin
from .element import ElementPlot
class HVSpanPlot(ShapePlot):
    apply_ranges = param.Boolean(default=False, doc='\n        Whether to include the annotation in axis range calculations.')
    _shape_type = 'rect'
    _supports_geo = False

    def get_data(self, element, ranges, style, **kwargs):
        if isinstance(element, HSpan) and self.invert_axes or (isinstance(element, VSpan) and (not self.invert_axes)):
            x0, x1 = element.data
            visible = not (x0 is None and x1 is None)
            return [dict(x0=x0, x1=x1, y0=0, y1=1, xref='x', yref='paper', visible=visible)]
        else:
            y0, y1 = element.data
            visible = not (y0 is None and y1 is None)
            return [dict(x0=0.0, x1=1.0, y0=y0, y1=y1, xref='paper', yref='y', visible=visible)]