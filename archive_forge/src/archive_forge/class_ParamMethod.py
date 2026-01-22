from __future__ import annotations
import asyncio
import inspect
import itertools
import json
import os
import sys
import textwrap
import types
from collections import defaultdict, namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import (
import param
from param.parameterized import (
from param.reactive import rx
from .config import config
from .io import state
from .layout import (
from .pane import DataFrame as DataFramePane
from .pane.base import PaneBase, ReplacementPane
from .reactive import Reactive
from .util import (
from .util.checks import is_dataframe, is_mpl_axes, is_series
from .viewable import Layoutable, Viewable
from .widgets import (
from .widgets.button import _ButtonBase
class ParamMethod(ParamRef):
    """
    ParamMethod panes wrap methods on parameterized classes and
    rerenders the plot when any of the method's parameters change. By
    default ParamMethod will watch all parameters on the class owning
    the method or can be restricted to certain parameters by annotating
    the method using the param.depends decorator. The method may
    return any object which itself can be rendered as a Pane.
    """
    priority: ClassVar[float | bool | None] = 0.5

    @param.depends('object', watch=True)
    def _validate_object(self):
        dependencies = getattr(self.object, '_dinfo', {})
        if not dependencies or not dependencies.get('watch'):
            return
        self.param.warning("The method supplied for Panel to display was declared with `watch=True`, which will cause the method to be called twice for any change in a dependent Parameter. `watch` should be False when Panel is responsible for displaying the result of the method call, while `watch=True` should be reserved for methods that work via side-effects, e.g. by modifying internal state of a class or global state in an application's namespace.")

    def _link_object_params(self):
        parameterized = get_method_owner(self.object)
        params = parameterized.param.method_dependencies(self.object.__name__)
        deps = params

        def update_pane(*events):
            if any((is_parameterized(event.new) for event in events)):
                new_deps = parameterized.param.method_dependencies(self.object.__name__)
                for p in list(deps):
                    if p in new_deps:
                        continue
                    watchers = self._internal_callbacks
                    for w in list(watchers):
                        if w.inst is p.inst and w.cls is p.cls and (p.name in w.parameter_names):
                            obj = p.cls if p.inst is None else p.inst
                            obj.param.unwatch(w)
                            watchers.remove(w)
                    deps.remove(p)
                new_deps = [dep for dep in new_deps if dep not in deps]
                for _, params in full_groupby(new_deps, lambda x: (x.inst or x.cls, x.what)):
                    p = params[0]
                    pobj = p.cls if p.inst is None else p.inst
                    ps = [_p.name for _p in params]
                    watcher = pobj.param.watch(update_pane, ps, p.what)
                    self._internal_callbacks.append(watcher)
                    for p in params:
                        deps.append(p)
            self._replace_pane()
        for _, sub_params in full_groupby(params, lambda x: (x.inst or x.cls, x.what)):
            p = sub_params[0]
            pobj = p.inst or p.cls
            ps = [_p.name for _p in sub_params]
            if isinstance(pobj, Reactive) and self.loading_indicator:
                props = {p: 'loading' for p in ps if p in pobj._linkable_params}
                if props:
                    pobj.jslink(self._inner_layout, **props)
            watcher = pobj.param.watch(update_pane, ps, p.what)
            self._internal_callbacks.append(watcher)

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        return inspect.ismethod(obj) and isinstance(get_method_owner(obj), param.Parameterized)

    @classmethod
    def eval(cls, ref):
        return eval_function_with_deps(ref)