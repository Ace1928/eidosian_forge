import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
def _compute_tick_mapping(self, kind, order, bins):
    """
        Helper function to compute tick mappings based on `ticks` and
        default orders and bins.
        """
    if kind == 'angle':
        ticks = self.xticks
        reverse = True
    elif kind == 'radius':
        ticks = self.yticks
        reverse = False
    if callable(ticks):
        text_nth = [x for x in order if ticks(x)]
    elif isinstance(ticks, (tuple, list)):
        bins = self._get_bins(kind, ticks, reverse)
        text_nth = ticks
    elif ticks:
        nth_label = np.ceil(len(order) / float(ticks)).astype(int)
        text_nth = order[::nth_label]
    return {x: bins[x] for x in text_nth}