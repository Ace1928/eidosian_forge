from collections import defaultdict
import numpy as np
import param
from ...core import util
from ...element import Contours, Polygons
from ...util.transform import dim
from .callbacks import PolyDrawCallback, PolyEditCallback
from .element import ColorbarPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import multi_polygons_data
class PolygonPlot(ContourPlot):
    style_opts = base_properties + line_properties + fill_properties + ['cmap']
    _plot_methods = dict(single='patches', batched='patches')
    _batched_style_opts = line_properties + fill_properties
    _color_style = 'fill_color'
    selection_display = BokehOverlaySelectionDisplay()