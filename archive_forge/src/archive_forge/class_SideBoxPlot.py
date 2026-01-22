import numpy as np
import param
from ...core.ndmapping import sorted_context
from ..mixins import MultiDistributionMixin
from .chart import AreaPlot, ChartPlot
from .path import PolygonPlot
from .plot import AdjoinedPlot
class SideBoxPlot(AdjoinedPlot, BoxPlot):
    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc='\n        Make plot background invisible.')
    border_size = param.Number(default=0, doc='\n        The size of the border expressed as a fraction of the main plot.')
    xaxis = param.ObjectSelector(default='bare', objects=['top', 'bottom', 'bare', 'top-bare', 'bottom-bare', None], doc="\n        Whether and where to display the xaxis, bare options allow suppressing\n        all axis labels including ticks and xlabel. Valid options are 'top',\n        'bottom', 'bare', 'top-bare' and 'bottom-bare'.")
    yaxis = param.ObjectSelector(default='bare', objects=['left', 'right', 'bare', 'left-bare', 'right-bare', None], doc="\n        Whether and where to display the yaxis, bare options allow suppressing\n        all axis labels including ticks and ylabel. Valid options are 'left',\n        'right', 'bare' 'left-bare' and 'right-bare'.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.adjoined:
            self.invert_axes = not self.invert_axes