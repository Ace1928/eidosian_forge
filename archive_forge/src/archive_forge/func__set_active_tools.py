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
def _set_active_tools(self, plot):
    """Activates the list of active tools"""
    if plot is None or self.toolbar == 'disable':
        return
    if self.active_tools is None:
        enabled_tools = set(self.default_tools + self.tools)
        active_tools = {'pan', 'wheel_zoom'} & enabled_tools
    else:
        active_tools = self.active_tools
    if active_tools == []:
        plot.toolbar.active_drag = None
    for tool in active_tools:
        if isinstance(tool, str):
            tool_type = TOOL_TYPES.get(tool, type(None))
            matching = [t for t in plot.toolbar.tools if isinstance(t, tool_type)]
            if not matching:
                self.param.warning(f'Tool of type {tool!r} could not be found and could not be activated by default.')
                continue
            tool = matching[0]
        if isinstance(tool, tools.Drag):
            plot.toolbar.active_drag = tool
        if isinstance(tool, tools.Scroll):
            plot.toolbar.active_scroll = tool
        if isinstance(tool, tools.Tap):
            plot.toolbar.active_tap = tool
        if isinstance(tool, tools.InspectTool):
            plot.toolbar.active_inspect.append(tool)