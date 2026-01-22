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
class CompositeElementPlot(ElementPlot):
    """
    A CompositeElementPlot is an Element plot type that coordinates
    drawing of multiple glyphs.
    """
    _style_groups = {}
    _draw_order = []

    def _init_glyphs(self, plot, element, ranges, source, data=None, mapping=None, style=None):
        if None in (data, mapping):
            style = self.style[self.cyclic_index]
            data, mapping, style = self.get_data(element, ranges, style)
        keys = glyph_order(dict(data, **mapping), self._draw_order)
        source_cache = {}
        current_id = element._plot_id
        self.handles['previous_id'] = current_id
        for key in keys:
            style_group = self._style_groups.get('_'.join(key.split('_')[:-1]))
            group_style = dict(style)
            ds_data = data.get(key, {})
            with abbreviated_exception():
                group_style = self._apply_transforms(element, ds_data, ranges, group_style, style_group)
            if id(ds_data) in source_cache:
                source = source_cache[id(ds_data)]
            else:
                source = self._init_datasource(ds_data)
                source_cache[id(ds_data)] = source
            self.handles[key + '_source'] = source
            properties = self._glyph_properties(plot, element, source, ranges, group_style, style_group)
            properties = self._process_properties(key, properties, mapping.get(key, {}))
            with abbreviated_exception():
                renderer, glyph = self._init_glyph(plot, mapping.get(key, {}), properties, key)
            self.handles[key + '_glyph'] = glyph
            if isinstance(renderer, Renderer):
                self.handles[key + '_glyph_renderer'] = renderer
            self._postprocess_hover(renderer, source)
            with abbreviated_exception():
                self._update_glyph(renderer, properties, mapping.get(key, {}), glyph, source, source.data)
        if getattr(self, 'colorbar', False):
            for k, v in list(self.handles.items()):
                if not k.endswith('color_mapper'):
                    continue
                self._draw_colorbar(plot, v, k.replace('color_mapper', ''))

    def _process_properties(self, key, properties, mapping):
        key = '_'.join(key.split('_')[:-1]) if '_' in key else key
        style_group = self._style_groups[key]
        group_props = {}
        for k, v in properties.items():
            if k in self.style_opts:
                group = k.split('_')[0]
                if group == style_group:
                    if k in mapping:
                        v = mapping[k]
                    k = '_'.join(k.split('_')[1:])
                else:
                    continue
            group_props[k] = v
        return group_props

    def _update_glyphs(self, element, ranges, style):
        plot = self.handles['plot']
        previous_id = self.handles.get('previous_id', None)
        if self.batched:
            current_id = tuple(element.traverse(lambda x: x._plot_id, [Element]))
        else:
            current_id = element._plot_id
        self.handles['previous_id'] = current_id
        self.static_source = self.dynamic and current_id == previous_id
        data, mapping, style = self.get_data(element, ranges, style)
        keys = glyph_order(dict(data, **mapping), self._draw_order)
        for key in keys:
            gdata = data.get(key)
            source = self.handles[key + '_source']
            glyph = self.handles.get(key + '_glyph')
            if glyph:
                group_style = dict(style)
                style_group = self._style_groups.get('_'.join(key.split('_')[:-1]))
                with abbreviated_exception():
                    group_style = self._apply_transforms(element, gdata, ranges, group_style, style_group)
                properties = self._glyph_properties(plot, element, source, ranges, group_style, style_group)
                properties = self._process_properties(key, properties, mapping[key])
                renderer = self.handles.get(key + '_glyph_renderer')
                with abbreviated_exception():
                    self._update_glyph(renderer, properties, mapping[key], glyph, source, gdata)
            elif not self.static_source and gdata is not None:
                self._update_datasource(source, gdata)

    def _init_glyph(self, plot, mapping, properties, key):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        plot_method = '_'.join(key.split('_')[:-1])
        renderer = getattr(plot, plot_method)(**dict(properties, **mapping))
        return (renderer, renderer.glyph)