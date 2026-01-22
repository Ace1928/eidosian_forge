from __future__ import annotations
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ImportedStyleSheet
from bokeh.models.layouts import (
from .._param import Margin
from ..io.cache import _generate_hash
from ..io.document import create_doc_if_none_exists, unlocked
from ..io.notebook import push
from ..io.state import state
from ..layout.base import (
from ..links import Link
from ..models import ReactiveHTML as _BkReactiveHTML
from ..reactive import Reactive
from ..util import param_reprs, param_watchers
from ..util.checks import is_dataframe, is_series
from ..util.parameters import get_params_to_inherit
from ..viewable import (
class ReplacementPane(PaneBase):
    """
    ReplacementPane provides a baseclass for dynamic components that
    may have to dynamically update or switch out their contents, e.g.
    a dynamic callback that may return different objects to be rendered.

    When the pane updates it either entirely replaces the underlying
    `bokeh.model.Model`, by creating an internal layout to replace the
    children on, or updates the existing model in place.
    """
    inplace = param.Boolean(default=False, doc='\n        Whether to update the object inplace.')
    object = param.Parameter(default=None, allow_refs=False, doc='\n        The object being wrapped, which will be converted to a\n        Bokeh model.')
    _pane = param.ClassSelector(class_=Viewable, allow_refs=False)
    _ignored_refs: ClassVar[Tuple[str, ...]] = ('object',)
    _linked_properties: ClassVar[Tuple[str, ...]] = ()
    _rename: ClassVar[Mapping[str, str | None]] = {'_pane': None, 'inplace': None}
    _updates: bool = True
    __abstract = True

    def __init__(self, object: Any=None, **params):
        self._kwargs = {p: params.pop(p) for p in list(params) if p not in self.param}
        super().__init__(object, **params)
        self._pane = panel(None)
        self._internal = True
        self._inner_layout = Column(self._pane, **{k: v for k, v in params.items() if k in Column.param})
        self._internal_callbacks.append(self.param.watch(self._update_inner_layout, list(Layoutable.param)))
        self._sync_layout()

    def _get_model(self, doc: Document, root: Model | None=None, parent: Model | None=None, comm: Comm | None=None) -> Model:
        if root:
            ref = root.ref['id']
            if ref in self._models:
                self._cleanup(root)
        model = self._inner_layout._get_model(doc, root, parent, comm)
        root = root or model
        self._models[root.ref['id']] = (model, parent)
        return model

    @param.depends('_pane', '_pane.sizing_mode', '_pane.width_policy', '_pane.height_policy', watch=True)
    def _sync_layout(self):
        if not hasattr(self, '_inner_layout') or (self._pane is not None and getattr(self._pane, '_object_changing', False)):
            return
        self._inner_layout.param.update({k: v for k, v in self._pane.param.values().items() if k in ('sizing_mode', 'width_policy', 'height_policy')})

    def _update_inner_layout(self, *events):
        self._pane.param.update({event.name: event.new for event in events})

    @classmethod
    def _recursive_update(cls, old: Reactive, new: Reactive):
        """
        Recursively descends through Panel layouts and diffs their
        contents updating only changed parameters ensuring we don't
        have to trigger a full re-render of the entire component.

        Arguments
        ---------
        old: Reactive
          The Reactive component being updated or replaced.
        new: Reactive
          The new Reactive component that the old one is being updated
          or replaced with.
        """
        ignored = ('name',)
        if isinstance(new, ListPanel):
            if len(old) == len(new):
                for i, (sub_old, sub_new) in enumerate(zip(old, new)):
                    if type(sub_old) is not type(sub_new):
                        old[i] = new
                        continue
                    if isinstance(new, NamedListPanel):
                        old._names[i] = new._names[i]
                    cls._recursive_update(sub_old, sub_new)
                ignored += ('objects',)
        pvals = dict(old.param.values())
        new_params = {}
        for p, p_new in new.param.values().items():
            p_old = pvals[p]
            if p in ignored or p_new is p_old:
                continue
            try:
                equal = p_new == p_old
                if is_dataframe(equal) or is_series(equal) or isinstance(equal, np.ndarray):
                    equal = equal.all()
                equal = bool(equal)
            except Exception:
                try:
                    equal = _generate_hash(p_new) == _generate_hash(p_old)
                except Exception:
                    equal = False
            if not equal:
                new_params[p] = p_new
        if isinstance(old, PaneBase):
            changing = any((p in old._rerender_params for p in new_params))
            old._object_changing = changing
            try:
                with param.edit_constant(old):
                    old.param.update(**new_params)
            finally:
                old._object_changing = False
        else:
            with param.edit_constant(old):
                old.param.update(**new_params)

    @classmethod
    def _update_from_object(cls, object: Any, old_object: Any, was_internal: bool, inplace: bool=False, **kwargs):
        pane_type = cls.get_pane_type(object)
        try:
            links = Link.registry.get(object)
        except TypeError:
            links = []
        custom_watchers = []
        if isinstance(object, Reactive):
            watchers = [w for pwatchers in param_watchers(object).values() for awatchers in pwatchers.values() for w in awatchers]
            custom_watchers = [wfn for wfn in watchers if wfn not in object._internal_callbacks and (not hasattr(wfn.fn, '_watcher_name'))]
        pane, internal = (None, was_internal)
        if type(old_object) is pane_type and (not links and (not custom_watchers) and was_internal or inplace):
            if isinstance(object, Panel) and len(old_object) == len(object):
                for i, (old, new) in enumerate(zip(old_object, object)):
                    if type(old) is not type(new):
                        old_object[i] = new
                        continue
                    cls._recursive_update(old, new)
            elif isinstance(object, Reactive):
                cls._recursive_update(old_object, object)
            elif old_object.object is not object:
                old_object.object = object
        else:
            pane_params = {k: v for k, v in kwargs.items() if k in pane_type.param}
            pane = panel(object, **pane_params)
            internal = pane is not object
        return (pane, internal)

    def _update_inner(self, new_object: Any) -> None:
        kwargs = dict(self.param.values(), **self._kwargs)
        del kwargs['object']
        new_pane, internal = self._update_from_object(new_object, self._pane, self._internal, **kwargs)
        if new_pane is None:
            return
        self._pane = new_pane
        self._inner_layout[:] = [self._pane]
        self._internal = internal

    def _cleanup(self, root: Model | None=None) -> None:
        self._inner_layout._cleanup(root)
        super()._cleanup(root)

    def _update_pane(self, *events) -> None:
        """
        Updating of the object should be handled manually.
        """

    def select(self, selector: type | Callable | None=None) -> List[Viewable]:
        """
        Iterates over the Viewable and any potential children in the
        applying the Selector.

        Arguments
        ---------
        selector: (type | callable | None)
          The selector allows selecting a subset of Viewables by
          declaring a type or callable function to filter by.

        Returns
        -------
        viewables: list(Viewable)
        """
        selected = super().select(selector)
        selected += self._inner_layout.select(selector)
        return selected