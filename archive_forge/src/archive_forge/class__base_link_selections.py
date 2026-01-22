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
class _base_link_selections(param.ParameterizedFunction):
    """
    Baseclass for linked selection functions.

    Subclasses override the _build_selection_streams class method to construct
    a _SelectionStreams namedtuple instance that includes the required streams
    for implementing linked selections.

    Subclasses also override the _expr_stream_updated method. This allows
    subclasses to control whether new selections override prior selections or
    whether they are combined with prior selections
    """
    link_inputs = param.Boolean(default=False, doc='\n        Whether to link any streams on the input to the output.')
    show_regions = param.Boolean(default=True, doc='\n        Whether to highlight the selected regions.')

    @bothmethod
    def instance(self_or_cls, **params):
        inst = super().instance(**params)
        inst._cross_filter_stream = CrossFilterSet(mode=inst.cross_filter_mode)
        inst._selection_override = _SelectionExprOverride()
        inst._selection_expr_streams = {}
        inst._plot_reset_streams = {}
        inst._selection_streams = self_or_cls._build_selection_streams(inst)
        return inst

    def _update_mode(self, event):
        if event.new == 'replace':
            self.selection_mode = 'overwrite'
        elif event.new == 'append':
            self.selection_mode = 'union'
        elif event.new == 'intersect':
            self.selection_mode = 'intersect'
        elif event.new == 'subtract':
            self.selection_mode = 'inverse'

    def _register(self, hvobj):
        """
        Register an Element or DynamicMap that may be capable of generating
        selection expressions in response to user interaction events
        """
        from .element import Table
        selection_expr_seq = SelectionExprSequence(hvobj, mode=self.selection_mode, include_region=self.show_regions, index_cols=self.index_cols)
        self._selection_expr_streams[hvobj] = selection_expr_seq
        self._cross_filter_stream.append_input_stream(selection_expr_seq)
        self._plot_reset_streams[hvobj] = PlotReset(source=hvobj)

        def clear_stream_history(resetting, stream=selection_expr_seq.history_stream):
            if resetting:
                stream.clear_history()
                stream.event()
        if not isinstance(hvobj, Table):
            mode_stream = SelectMode(source=hvobj)
            mode_stream.param.watch(self._update_mode, 'mode')
        self._plot_reset_streams[hvobj].param.watch(clear_stream_history, ['resetting'])

    def __call__(self, hvobj, **kwargs):
        self.param.update(**kwargs)
        if Store.current_backend not in Store.renderers:
            raise RuntimeError('Cannot perform link_selections operation since the selected backend %r is not loaded. Load the plotting extension with hv.extension or import the plotting backend explicitly.' % Store.current_backend)
        return self._selection_transform(hvobj.clone())

    def _selection_transform(self, hvobj, operations=()):
        """
        Transform an input HoloViews object into a dynamic object with linked
        selections enabled.
        """
        from .plotting.util import initialize_dynamic
        if isinstance(hvobj, DynamicMap):
            callback = hvobj.callback
            if len(callback.inputs) > 1:
                return Overlay([self._selection_transform(el) for el in callback.inputs]).collate()
            initialize_dynamic(hvobj)
            if issubclass(hvobj.type, Element):
                self._register(hvobj)
                chart = Store.registry[Store.current_backend][hvobj.type]
                return chart.selection_display(hvobj).build_selection(self._selection_streams, hvobj, operations, self._selection_expr_streams.get(hvobj, None), cache=self._cache)
            elif issubclass(hvobj.type, Overlay) and getattr(hvobj.callback, 'name', None) == 'dynamic_mul':
                return Overlay([self._selection_transform(el, operations=operations) for el in callback.inputs]).collate()
            elif getattr(hvobj.callback, 'name', None) == 'dynamic_operation':
                obj = callback.inputs[0]
                return self._selection_transform(obj, operations=operations).apply(callback.operation)
            else:
                self.param.warning(f"linked selection: Encountered DynamicMap that we don't know how to recurse into:\n{hvobj!r}")
                return hvobj
        elif isinstance(hvobj, Element):
            chart = Store.registry[Store.current_backend][type(hvobj)]
            if getattr(chart, 'selection_display', None) is not None:
                element = hvobj.clone(link=self.link_inputs)
                self._register(element)
                return chart.selection_display(element).build_selection(self._selection_streams, element, operations, self._selection_expr_streams.get(element, None), cache=self._cache)
            return hvobj
        elif isinstance(hvobj, (Layout, Overlay, NdOverlay, GridSpace, AdjointLayout)):
            data = dict([(k, self._selection_transform(v, operations)) for k, v in hvobj.items()])
            if isinstance(hvobj, NdOverlay):

                def compose(*args, **kwargs):
                    new = []
                    for k, v in data.items():
                        for i, el in enumerate(v[()]):
                            if i == len(new):
                                new.append([])
                            new[i].append((k, el))
                    return Overlay([hvobj.clone(n) for n in new])
                new_hvobj = DynamicMap(compose)
                new_hvobj.callback.inputs[:] = list(data.values())
            else:
                new_hvobj = hvobj.clone(data)
                if hasattr(new_hvobj, 'collate'):
                    new_hvobj = new_hvobj.collate()
            return new_hvobj
        else:
            return hvobj

    @classmethod
    def _build_selection_streams(cls, inst):
        """
        Subclasses should override this method to return a _SelectionStreams
        instance
        """
        raise NotImplementedError()

    def _expr_stream_updated(self, hvobj, selection_expr, bbox, region_element, **kwargs):
        """
        Called when one of the registered HoloViews objects produces a new
        selection expression.  Subclasses should override this method, and
        they should use the input expression to update the `exprs_stream`
        property of the _SelectionStreams instance that was produced by
        the _build_selection_streams.

        Subclasses have the flexibility to control whether the new selection
        express overrides previous selections, or whether it is combined with
        previous selections.
        """
        raise NotImplementedError()