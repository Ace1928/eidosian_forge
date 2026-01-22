import numpy as np
import param
from plotly import colors
from plotly.figure_factory._trisurf import trisurf as trisurface
from ...core.options import SkipRendering
from .chart import CurvePlot, ScatterPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class Scatter3DPlot(Chart3DPlot, ScatterPlot):
    style_opts = ['visible', 'marker', 'color', 'cmap', 'alpha', 'opacity', 'size', 'sizemin']
    _supports_geo = False
    selection_display = PlotlyOverlaySelectionDisplay(supports_region=False)

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'scatter3d', 'mode': 'markers'}