import param
from ..mixins import MultiDistributionMixin
from .chart import ChartPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class ViolinPlot(MultiDistributionPlot):
    box = param.Boolean(default=True, doc='\n        Whether to draw a boxplot inside the violin')
    meanline = param.Boolean(default=False, doc='\n        If "True", the mean of the box(es)\' underlying distribution\n        is drawn as a dashed line inside the box(es). If "sd" the\n        standard deviation is also drawn.')
    style_opts = ['visible', 'color', 'alpha', 'outliercolor', 'marker', 'size']
    _style_key = 'marker'

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'violin'}

    def graph_options(self, element, ranges, style, **kwargs):
        options = super().graph_options(element, ranges, style, **kwargs)
        options['meanline'] = {'visible': self.meanline}
        options['box'] = {'visible': self.box}
        return options