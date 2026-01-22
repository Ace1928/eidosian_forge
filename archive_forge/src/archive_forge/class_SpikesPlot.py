from collections import defaultdict
import numpy as np
import param
from bokeh.models import CategoricalColorMapper, CustomJS, FactorRange, Range1d, Whisker
from bokeh.models.tools import BoxSelectTool
from bokeh.transform import jitter
from ...core.data import Dataset
from ...core.dimension import dimension_name
from ...core.util import dimension_sanitizer, isfinite
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..util import compute_sizes, get_min_distance
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import categorize_array
class SpikesPlot(SpikesMixin, ColorbarPlot):
    spike_length = param.Number(default=0.5, doc='\n      The length of each spike if Spikes object is one dimensional.')
    position = param.Number(default=0.0, doc='\n      The position of the lower end of each spike.')
    show_legend = param.Boolean(default=True, doc='\n        Whether to show legend for the plot.')
    color_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of color style mapping, e.g. `color=dim('color')`")
    selection_display = BokehOverlaySelectionDisplay()
    style_opts = base_properties + line_properties + ['cmap', 'palette']
    _nonvectorized_styles = base_properties + ['cmap']
    _plot_methods = dict(single='segment')

    def get_data(self, element, ranges, style):
        dims = element.dimensions()
        data = {}
        pos = self.position
        opts = self.lookup_options(element, 'plot').options
        if len(element) == 0 or self.static_source:
            data = {'x': [], 'y0': [], 'y1': []}
        else:
            data['x'] = element.dimension_values(0)
            data['y0'] = np.full(len(element), pos)
            if len(dims) > 1 and 'spike_length' not in opts:
                data['y1'] = element.dimension_values(1) + pos
            else:
                data['y1'] = data['y0'] + self.spike_length
        if self.invert_axes:
            mapping = {'x0': 'y0', 'x1': 'y1', 'y0': 'x', 'y1': 'x'}
        else:
            mapping = {'x0': 'x', 'x1': 'x', 'y0': 'y0', 'y1': 'y1'}
        cdata, cmapping = self._get_color_data(element, ranges, dict(style))
        data.update(cdata)
        mapping.update(cmapping)
        self._get_hover_data(data, element)
        return (data, mapping, style)