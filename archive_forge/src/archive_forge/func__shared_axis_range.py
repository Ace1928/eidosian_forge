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
def _shared_axis_range(self, plots, specs, range_type, axis_type, pos):
    """
        Given a list of other plots return the shared axis from another
        plot by matching the dimensions specs stored as tags on the
        dimensions. Returns None if there is no such axis.
        """
    dim_range = None
    categorical = range_type is FactorRange
    for plot in plots:
        if plot is None or specs is None:
            continue
        ax = 'x' if pos == 0 else 'y'
        plot_range = getattr(plot, f'{ax}_range', None)
        axes = getattr(plot, f'{ax}axis', None)
        extra_ranges = getattr(plot, f'extra_{ax}_ranges', {})
        if plot_range and plot_range.tags and match_dim_specs(plot_range.tags[0], specs) and match_ax_type(axes[0], axis_type) and (not (categorical and (not isinstance(dim_range, FactorRange)))):
            dim_range = plot_range
        if dim_range is not None:
            break
        for extra_range in extra_ranges.values():
            if extra_range.tags and match_dim_specs(extra_range.tags[0], specs) and match_yaxis_type_to_range(axes, axis_type, extra_range.name) and (not (categorical and (not isinstance(dim_range, FactorRange)))):
                dim_range = extra_range
                break
    return dim_range