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
class ParamFunction(ParamRef):
    """
    ParamFunction panes wrap functions decorated with the param.depends
    decorator and rerenders the output when any of the function's
    dependencies change. This allows building reactive components into
    a Panel which depend on other parameters, e.g. tying the value of
    a widget to some other output.
    """
    priority: ClassVar[float | bool | None] = 0.6
    _applies_kw: ClassVar[bool] = True

    @param.depends('object', watch=True)
    def _validate_object(self):
        dependencies = getattr(self.object, '_dinfo', {})
        if not dependencies or not dependencies.get('watch'):
            return
        self.param.warning("The function supplied for Panel to display was declared with `watch=True`, which will cause the function to be called twice for any change in a dependent Parameter. `watch` should be False when Panel is responsible for displaying the result of the function call, while `watch=True` should be reserved for functions that work via side-effects, e.g. by modifying internal state of a class or global state in an application's namespace.")

    @classmethod
    def applies(cls, obj: Any, **kwargs) -> float | bool | None:
        if isinstance(obj, types.FunctionType):
            if hasattr(obj, '_dinfo'):
                return True
            if kwargs.get('defer_load') or cls.param.defer_load.default or (cls.param.defer_load.default is None and config.defer_load) or iscoroutinefunction(obj):
                return True
            return None
        return False

    @classmethod
    def eval(self, ref):
        return eval_function_with_deps(ref)