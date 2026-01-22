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
@property
def _subcoord_overlaid(self):
    """
        Indicates when the context is a subcoordinate plot, either from within
        the overlay rendering or one of its subplots. Used to skip code paths
        when rendering an element outside of an overlay.
        """
    if self._subcoord_standalone_ is not None:
        return self._subcoord_standalone_
    self._subcoord_standalone_ = isinstance(self, OverlayPlot) and self.subcoordinate_y or (not isinstance(self, OverlayPlot) and self.overlaid and self.subcoordinate_y)
    return self._subcoord_standalone_