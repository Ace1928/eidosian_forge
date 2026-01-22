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
def _init_tools(self, element, callbacks=None):
    """
        Processes the list of tools to be supplied to the plot.
        """
    if callbacks is None:
        callbacks = []
    hover_tools = {}
    init_tools, tool_types = ([], [])
    for key, subplot in self.subplots.items():
        el = element.get(key)
        if el is not None:
            el_tools = subplot._init_tools(el, self.callbacks)
            for tool in el_tools:
                if isinstance(tool, str):
                    tool_type = TOOL_TYPES.get(tool)
                else:
                    tool_type = type(tool)
                if isinstance(tool, tools.HoverTool):
                    tooltips = tuple(tool.tooltips) if tool.tooltips else ()
                    if tooltips in hover_tools:
                        continue
                    else:
                        hover_tools[tooltips] = tool
                elif tool_type in tool_types:
                    continue
                else:
                    tool_types.append(tool_type)
                init_tools.append(tool)
    self.handles['hover_tools'] = hover_tools
    return init_tools