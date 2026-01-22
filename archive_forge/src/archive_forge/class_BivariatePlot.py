import param
from ..mixins import MultiDistributionMixin
from .chart import ChartPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class BivariatePlot(ChartPlot, ColorbarPlot):
    filled = param.Boolean(default=False)
    ncontours = param.Integer(default=None)
    style_opts = ['visible', 'cmap', 'showlabels', 'labelfont', 'labelformat', 'showlines']
    _style_key = 'contours'
    selection_display = PlotlyOverlaySelectionDisplay()

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'histogram2dcontour'}

    def graph_options(self, element, ranges, style, **kwargs):
        opts = super().graph_options(element, ranges, style, **kwargs)
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        if self.ncontours:
            opts['autocontour'] = False
            opts['ncontours'] = self.ncontours
        opts['line'] = {'width': 1}
        opts['contours'] = {'coloring': 'fill' if self.filled else 'lines', 'showlines': style.get('showlines', True)}
        opts['colorscale'] = copts['colorscale']
        if 'colorbar' in copts:
            opts['colorbar'] = copts['colorbar']
        opts['showscale'] = copts.get('showscale', False)
        opts['visible'] = style.get('visible', True)
        return opts