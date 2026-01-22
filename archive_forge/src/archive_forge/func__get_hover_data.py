import warnings
from itertools import chain
from types import FunctionType
import bokeh
import bokeh.plotting
import numpy as np
import param
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.models.mappers import (
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.scales import LogScale
from bokeh.models.tickers import (
from bokeh.models.tools import Tool
from packaging.version import Version
from ...core import CompositeOverlay, Dataset, Dimension, DynamicMap, Element, util
from ...core.options import Keywords, SkipRendering, abbreviated_exception
from ...element import Annotation, Contours, Graph, Path, Tiles, VectorField
from ...streams import Buffer, PlotSize, RangeXY
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_axis_label, dim_range_key, process_cmap
from .plot import BokehPlot
from .styles import (
from .tabular import TablePlot
from .util import (
def _get_hover_data(self, data, element, dimensions=None):
    """
        Initializes hover data based on Element dimension values.
        If empty initializes with no data.
        """
    if 'hover' not in self.handles or self.static_source:
        return
    for d in dimensions or element.dimensions():
        dim = util.dimension_sanitizer(d.name)
        if dim not in data:
            data[dim] = element.dimension_values(d)
    for k, v in self.overlay_dims.items():
        dim = util.dimension_sanitizer(k.name)
        if dim not in data:
            data[dim] = [v] * len(next(iter(data.values())))