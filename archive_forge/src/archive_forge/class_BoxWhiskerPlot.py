import param
from ..mixins import MultiDistributionMixin
from .chart import ChartPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class BoxWhiskerPlot(MultiDistributionPlot):
    boxpoints = param.ObjectSelector(objects=['all', 'outliers', 'suspectedoutliers', False], default='outliers', doc="\n        Which points to show, valid options are 'all', 'outliers',\n        'suspectedoutliers' and False")
    jitter = param.Number(default=0, doc='\n        Sets the amount of jitter in the sample points drawn. If "0",\n        the sample points align along the distribution axis. If "1",\n        the sample points are drawn in a random jitter of width equal\n        to the width of the box(es).')
    mean = param.ObjectSelector(default=False, objects=[True, False, 'sd'], doc='\n        If "True", the mean of the box(es)\' underlying distribution\n        is drawn as a dashed line inside the box(es). If "sd" the\n        standard deviation is also drawn.')
    style_opts = ['visible', 'color', 'alpha', 'outliercolor', 'marker', 'size']
    _style_key = 'marker'
    selection_display = PlotlyOverlaySelectionDisplay()

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'box'}

    def graph_options(self, element, ranges, style, **kwargs):
        options = super().graph_options(element, ranges, style, **kwargs)
        options['boxmean'] = self.mean
        options['jitter'] = self.jitter
        return options