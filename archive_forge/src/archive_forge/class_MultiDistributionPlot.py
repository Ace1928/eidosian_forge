import param
from ..mixins import MultiDistributionMixin
from .chart import ChartPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class MultiDistributionPlot(MultiDistributionMixin, ElementPlot):

    def get_data(self, element, ranges, style, **kwargs):
        if element.kdims:
            groups = element.groupby(element.kdims).items()
        else:
            groups = [(element.label, element)]
        plots = []
        axis = 'x' if self.invert_axes else 'y'
        for key, group in groups:
            if element.kdims:
                label = ','.join([d.pprint_value(v) for d, v in zip(element.kdims, key)])
            else:
                label = key
            data = {axis: group.dimension_values(group.vdims[0]), 'name': label}
            plots.append(data)
        return plots