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
def _create_extra_axes(self, plots, subplots, element, ranges):
    if self.invert_axes:
        axpos0, axpos1 = ('below', 'above')
    else:
        axpos0, axpos1 = ('left', 'right')
    ax_specs, yaxes, dimensions = ({}, {}, {})
    subcoordinate_axes = 0
    for el, sp in zip(element, self.subplots.values()):
        ax_dims = sp._get_axis_dims(el)[:2]
        if sp.invert_axes:
            ax_dims[::-1]
        yd = ax_dims[1]
        opts = el.opts.get('plot', backend='bokeh').kwargs
        if not isinstance(yd, Dimension) or yd.name in yaxes:
            continue
        if self._subcoord_overlaid:
            if opts.get('subcoordinate_y') is None:
                continue
            ax_name = el.label
            subcoordinate_axes += 1
        else:
            ax_name = yd.name
        dimensions[ax_name] = yd
        yaxes[ax_name] = {'position': opts.get('yaxis', axpos1 if len(yaxes) else axpos0), 'autorange': opts.get('autorange', None), 'logx': opts.get('logx', False), 'logy': opts.get('logy', False), 'invert_yaxis': opts.get('invert_yaxis', False), 'ylim': opts.get('ylim', (np.nan, np.nan)), 'label': opts.get('ylabel', dim_axis_label(yd)), 'fontsize': {'axis_label_text_font_size': sp._fontsize('ylabel').get('fontsize'), 'major_label_text_font_size': sp._fontsize('yticks').get('fontsize')}, 'subcoordinate_y': subcoordinate_axes - 1 if self._subcoord_overlaid else None}
    for ydim, info in yaxes.items():
        range_tags_extras = {'invert_yaxis': info['invert_yaxis']}
        if info['subcoordinate_y'] is not None:
            range_tags_extras['subcoordinate_y'] = info['subcoordinate_y']
        if info['autorange'] == 'y':
            range_tags_extras['autorange'] = True
            lowerlim, upperlim = (info['ylim'][0], info['ylim'][1])
            if not (lowerlim is None or np.isnan(lowerlim)):
                range_tags_extras['y-lowerlim'] = lowerlim
            if not (upperlim is None or np.isnan(upperlim)):
                range_tags_extras['y-upperlim'] = upperlim
        else:
            range_tags_extras['autorange'] = False
        ax_props = self._axis_props(plots, subplots, element, ranges, pos=1, dim=dimensions[ydim], range_tags_extras=range_tags_extras, extra_range_name=ydim)
        log_enabled = info['logx'] if self.invert_axes else info['logy']
        ax_type = 'log' if log_enabled else ax_props[0]
        ax_specs[ydim] = (ax_type, info['label'], ax_props[2], info['position'], info['fontsize'])
    return (yaxes, ax_specs)