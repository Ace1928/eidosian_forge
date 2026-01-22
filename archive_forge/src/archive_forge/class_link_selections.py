from collections import namedtuple
import numpy as np
import param
from param.parameterized import bothmethod
from .core.data import Dataset
from .core.element import Element, Layout
from .core.layout import AdjointLayout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import (
from .util import DynamicMap
class link_selections(_base_link_selections):
    """
    Operation which automatically links selections between elements
    in the supplied HoloViews object. Can be used a single time or
    be used as an instance to apply the linked selections across
    multiple objects.
    """
    cross_filter_mode = param.Selector(objects=['overwrite', 'intersect'], default='intersect', doc='\n        Determines how to combine selections across different\n        elements.')
    index_cols = param.List(default=None, doc='\n        If provided, selection switches to index mode where all queries\n        are expressed solely in terms of discrete values along the\n        index_cols.  All Elements given to link_selections must define the index_cols, either as explicit dimensions or by sharing an underlying Dataset that defines them.')
    selection_expr = param.Parameter(default=None, doc='\n        dim expression of the current selection or None to indicate\n        that everything is selected.')
    selected_color = param.Color(default=None, allow_None=True, doc='\n        Color of selected data, or None to use the original color of\n        each element.')
    selection_mode = param.Selector(objects=['overwrite', 'intersect', 'union', 'inverse'], default='overwrite', doc='\n        Determines how to combine successive selections on the same\n        element.')
    unselected_alpha = param.Magnitude(default=0.1, doc='\n        Alpha of unselected data.')
    unselected_color = param.Color(default=None, doc='\n        Color of unselected data.')

    @bothmethod
    def instance(self_or_cls, **params):
        inst = super().instance(**params)
        inst._obj_selections = {}
        inst._obj_regions = {}
        inst._reset_regions = True
        inst._datasets = []
        inst._cache = {}
        self_or_cls._install_param_callbacks(inst)
        return inst

    @param.depends('selection_expr', watch=True)
    def _update_pipes(self):
        sel_expr = self.selection_expr
        for pipe, ds, raw in self._datasets:
            ref = ds._plot_id
            self._cache[ref] = ds_cache = self._cache.get(ref, {})
            if sel_expr in ds_cache:
                data = ds_cache[sel_expr]
                return pipe.event(data=data.data)
            else:
                ds_cache.clear()
            sel_ds = SelectionDisplay._select(ds, sel_expr, self._cache)
            ds_cache[sel_expr] = sel_ds
            pipe.event(data=sel_ds.data if raw else sel_ds)

    def selection_param(self, data):
        """
        Returns a parameter which reflects the current selection
        when applied to the supplied data, making it easy to create
        a callback which depends on the current selection.

        Args:
            data: A Dataset type or data which can be cast to a Dataset

        Returns:
            A parameter which reflects the current selection
        """
        raw = False
        if not isinstance(data, Dataset):
            raw = True
            data = Dataset(data)
        pipe = Pipe(data=data.data)
        self._datasets.append((pipe, data, raw))
        return pipe.param.data

    def filter(self, data, selection_expr=None):
        """
        Filters the provided data based on the current state of the
        current selection expression.

        Args:
            data: A Dataset type or data which can be cast to a Dataset
            selection_expr: Optionally provide your own selection expression

        Returns:
            The filtered data
        """
        expr = self.selection_expr if selection_expr is None else selection_expr
        if expr is None:
            return data
        is_dataset = isinstance(data, Dataset)
        if not is_dataset:
            data = Dataset(data)
        filtered = data[expr.apply(data)]
        return filtered if is_dataset else filtered.data

    @bothmethod
    def _install_param_callbacks(self_or_cls, inst):

        def update_selection_mode(*_):
            for stream in inst._selection_expr_streams.values():
                stream.reset()
                stream.mode = inst.selection_mode
        inst.param.watch(update_selection_mode, ['selection_mode'])

        def update_cross_filter_mode(*_):
            inst._cross_filter_stream.reset()
            inst._cross_filter_stream.mode = inst.cross_filter_mode
        inst.param.watch(update_cross_filter_mode, ['cross_filter_mode'])

        def update_show_region(*_):
            for stream in inst._selection_expr_streams.values():
                stream.include_region = inst.show_regions
                stream.event()
        inst.param.watch(update_show_region, ['show_regions'])

        def update_selection_expr(*_):
            new_selection_expr = inst.selection_expr
            current_selection_expr = inst._cross_filter_stream.selection_expr
            if repr(new_selection_expr) != repr(current_selection_expr):
                if inst.show_regions:
                    inst.show_regions = False
                inst._selection_override.event(selection_expr=new_selection_expr)
        inst.param.watch(update_selection_expr, ['selection_expr'])

        def selection_expr_changed(*_):
            new_selection_expr = inst._cross_filter_stream.selection_expr
            if repr(inst.selection_expr) != repr(new_selection_expr):
                inst.selection_expr = new_selection_expr
        inst._cross_filter_stream.param.watch(selection_expr_changed, ['selection_expr'])
        for stream in inst._selection_expr_streams.values():

            def clear_stream_history(resetting, stream=stream):
                if resetting:
                    stream.clear_history()
            print('registering reset for ', stream)
            stream.plot_reset_stream.param.watch(clear_stream_history, ['resetting'])

    @classmethod
    def _build_selection_streams(cls, inst):
        style_stream = _Styles(colors=[inst.unselected_color, inst.selected_color], alpha=inst.unselected_alpha)
        cmap_streams = [_Cmap(cmap=inst.unselected_cmap), _Cmap(cmap=inst.selected_cmap)]

        def update_colors(*_):
            colors = [inst.unselected_color, inst.selected_color]
            style_stream.event(colors=colors, alpha=inst.unselected_alpha)
            cmap_streams[0].event(cmap=inst.unselected_cmap)
            if cmap_streams[1] is not None:
                cmap_streams[1].event(cmap=inst.selected_cmap)
        inst.param.watch(update_colors, ['unselected_color', 'selected_color', 'unselected_alpha'])
        exprs_stream = _SelectionExprLayers(inst._selection_override, inst._cross_filter_stream)
        return _SelectionStreams(style_stream=style_stream, exprs_stream=exprs_stream, cmap_streams=cmap_streams)

    @property
    def unselected_cmap(self):
        """
        The datashader colormap for unselected data
        """
        if self.unselected_color is None:
            return None
        return _color_to_cmap(self.unselected_color)

    @property
    def selected_cmap(self):
        """
        The datashader colormap for selected data
        """
        return None if self.selected_color is None else _color_to_cmap(self.selected_color)