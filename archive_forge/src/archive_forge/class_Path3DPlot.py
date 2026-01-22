import numpy as np
import param
from plotly import colors
from plotly.figure_factory._trisurf import trisurf as trisurface
from ...core.options import SkipRendering
from .chart import CurvePlot, ScatterPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class Path3DPlot(Chart3DPlot, CurvePlot):
    _per_trace = True
    _nonvectorized_styles = []

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'scatter3d', 'mode': 'lines'}

    def graph_options(self, element, ranges, style, **kwargs):
        opts = super().graph_options(element, ranges, style, **kwargs)
        opts['line'].pop('showscale', None)
        return opts

    def get_data(self, element, ranges, style, **kwargs):
        return [dict(x=el.dimension_values(0), y=el.dimension_values(1), z=el.dimension_values(2)) for el in element.split()]