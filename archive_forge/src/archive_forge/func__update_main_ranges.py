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
def _update_main_ranges(self, element, x_range, y_range, ranges):
    plot = self.handles['plot']
    l, b, r, t = (None, None, None, None)
    if any((isinstance(r, (Range1d, DataRange1d)) for r in [x_range, y_range])):
        if self.multi_y:
            range_dim = x_range.name if self.invert_axes else y_range.name
        else:
            range_dim = None
        try:
            l, b, r, t = self.get_extents(element, ranges, dimension=range_dim)
        except TypeError:
            l, b, r, t = self.get_extents(element, ranges)
        if self.invert_axes:
            l, b, r, t = (b, l, t, r)
    xfactors, yfactors = (None, None)
    if any((isinstance(ax_range, FactorRange) for ax_range in [x_range, y_range])):
        xfactors, yfactors = self._get_factors(element, ranges)
    framewise = self.framewise
    streaming = self.streaming and any((stream._triggering and stream.following for stream in self.streaming))
    xupdate = not (self.model_changed(x_range) or self.model_changed(plot)) and (framewise or streaming) or xfactors is not None
    yupdate = (not (self.model_changed(x_range) or self.model_changed(plot)) and (framewise or streaming) or yfactors is not None) and (not self.subcoordinate_y)
    options = self._traverse_options(element, 'plot', ['width', 'height'], defaults=False)
    fixed_width = self.frame_width or options.get('width')
    fixed_height = self.frame_height or options.get('height')
    constrained_width = options.get('min_width') or options.get('max_width')
    constrained_height = options.get('min_height') or options.get('max_height')
    data_aspect = self.aspect == 'equal' or self.data_aspect
    xaxis, yaxis = (self.handles['xaxis'], self.handles['yaxis'])
    categorical = isinstance(xaxis, CategoricalAxis) or isinstance(yaxis, CategoricalAxis)
    datetime = isinstance(xaxis, DatetimeAxis) or isinstance(yaxis, CategoricalAxis)
    range_streams = [s for s in self.streams if isinstance(s, RangeXY)]
    if data_aspect and (categorical or datetime):
        ax_type = 'categorical' if categorical else 'datetime axes'
        self.param.warning('Cannot set data_aspect if one or both axes are %s, the option will be ignored.' % ax_type)
    elif data_aspect:
        plot = self.handles['plot']
        xspan = r - l if util.is_number(l) and util.is_number(r) else None
        yspan = t - b if util.is_number(b) and util.is_number(t) else None
        if self.drawn or (fixed_width and fixed_height) or (constrained_width or constrained_height):
            ratio = self.data_aspect or 1
            if self.aspect == 'square':
                frame_aspect = 1
            elif self.aspect and self.aspect != 'equal':
                frame_aspect = self.aspect
            elif plot.frame_height and plot.frame_width:
                frame_aspect = plot.frame_height / plot.frame_width
            else:
                return
            if self.drawn:
                current_l, current_r = (plot.x_range.start, plot.x_range.end)
                current_b, current_t = (plot.y_range.start, plot.y_range.end)
                current_xspan, current_yspan = (current_r - current_l, current_t - current_b)
            else:
                current_l, current_r, current_b, current_t = (l, r, b, t)
                current_xspan, current_yspan = (xspan, yspan)
            if any((rs._triggering for rs in range_streams)):
                l, r, b, t = (current_l, current_r, current_b, current_t)
                xspan, yspan = (current_xspan, current_yspan)
            size_streams = [s for s in self.streams if isinstance(s, PlotSize)]
            if any((ss._triggering for ss in size_streams)) and self._updated:
                return
            desired_xspan = yspan * (ratio / frame_aspect)
            desired_yspan = xspan / (ratio / frame_aspect)
            if np.allclose(desired_xspan, xspan, rtol=0.05) and np.allclose(desired_yspan, yspan, rtol=0.05) or not (util.isfinite(xspan) and util.isfinite(yspan)):
                pass
            elif desired_yspan >= yspan:
                desired_yspan = current_xspan / (ratio / frame_aspect)
                ypad = (desired_yspan - yspan) / 2.0
                b, t = (b - ypad, t + ypad)
                yupdate = True
            else:
                desired_xspan = current_yspan * (ratio / frame_aspect)
                xpad = (desired_xspan - xspan) / 2.0
                l, r = (l - xpad, r + xpad)
                xupdate = True
        elif not (fixed_height and fixed_width):
            aspect = self.get_aspect(xspan, yspan)
            width = plot.frame_width or plot.width or 300
            height = plot.frame_height or plot.height or 300
            if not (fixed_width or fixed_height) and (not self.responsive):
                fixed_height = True
            if fixed_height:
                plot.frame_height = height
                plot.frame_width = int(height / aspect)
                plot.width, plot.height = (None, None)
            elif fixed_width:
                plot.frame_width = width
                plot.frame_height = int(width * aspect)
                plot.width, plot.height = (None, None)
            else:
                plot.aspect_ratio = 1.0 / aspect
        box_zoom = plot.select(type=tools.BoxZoomTool)
        scroll_zoom = plot.select(type=tools.WheelZoomTool)
        if box_zoom:
            box_zoom.match_aspect = True
        if scroll_zoom:
            scroll_zoom.zoom_on_axis = False
    elif any((rs._triggering for rs in range_streams)):
        xupdate, yupdate = (False, False)
    if not self.drawn or xupdate:
        self._update_range(x_range, l, r, xfactors, self.invert_xaxis, self._shared['x-main-range'], self.logx, streaming)
    if not (self.drawn or self.subcoordinate_y) or yupdate:
        self._update_range(y_range, b, t, yfactors, self._get_tag(y_range, 'invert_yaxis'), self._shared['y-main-range'], self.logy, streaming)