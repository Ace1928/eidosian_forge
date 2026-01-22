import numpy as np
import param
from ...element import Tiles
from ...operation import interpolate_curve
from ..mixins import AreaMixin, BarsMixin
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class CurvePlot(ChartPlot, ColorbarPlot):
    interpolation = param.ObjectSelector(objects=['linear', 'steps-mid', 'steps-pre', 'steps-post'], default='linear', doc="\n        Defines how the samples of the Curve are interpolated,\n        default is 'linear', other options include 'steps-mid',\n        'steps-pre' and 'steps-post'.")
    padding = param.ClassSelector(default=(0, 0.1), class_=(int, float, tuple))
    style_opts = ['visible', 'color', 'dash', 'line_width']
    _nonvectorized_styles = style_opts
    unsupported_geo_style_opts = ['dash']
    _style_key = 'line'
    _supports_geo = True

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        if is_geo:
            return {'type': 'scattermapbox', 'mode': 'lines'}
        else:
            return {'type': 'scatter', 'mode': 'lines'}

    def get_data(self, element, ranges, style, **kwargs):
        if 'steps' in self.interpolation:
            element = interpolate_curve(element, interpolation=self.interpolation)
        return super().get_data(element, ranges, style, **kwargs)