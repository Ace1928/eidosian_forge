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
def _draw_colorbar(self, plot, color_mapper, prefix=''):
    if CategoricalColorMapper and isinstance(color_mapper, CategoricalColorMapper):
        return
    if isinstance(color_mapper, EqHistColorMapper):
        ticker = BinnedTicker(mapper=color_mapper)
    elif isinstance(color_mapper, LogColorMapper) and color_mapper.low > 0:
        ticker = LogTicker()
    else:
        ticker = BasicTicker()
    cbar_opts = dict(self.colorbar_specs[self.colorbar_position])
    pos = cbar_opts['pos']
    if any((isinstance(model, ColorBar) for model in getattr(plot, pos, []))):
        return
    if self.clabel:
        self.colorbar_opts.update({'title': self.clabel})
    if self.cformatter is not None:
        self.colorbar_opts.update({'formatter': wrap_formatter(self.cformatter, 'c')})
    for tk in ['cticks', 'ticks']:
        ticksize = self._fontsize(tk, common=False).get('fontsize')
        if ticksize is not None:
            self.colorbar_opts.update({'major_label_text_font_size': ticksize})
            break
    for lb in ['clabel', 'labels']:
        labelsize = self._fontsize(lb, common=False).get('fontsize')
        if labelsize is not None:
            self.colorbar_opts.update({'title_text_font_size': labelsize})
            break
    opts = dict(cbar_opts['opts'], color_mapper=color_mapper, ticker=ticker, **self._colorbar_defaults)
    color_bar = ColorBar(**dict(opts, **self.colorbar_opts))
    plot.add_layout(color_bar, pos)
    self.handles[prefix + 'colorbar'] = color_bar