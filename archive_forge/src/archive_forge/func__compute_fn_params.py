from __future__ import annotations
import asyncio
import inspect
import math
import operator
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Optional
from .depends import depends
from .display import _display_accessors, _reactive_display_objs
from .parameterized import (
from .parameters import Boolean, Event
from ._utils import _to_async_gen, iscoroutinefunction, full_groupby
def _compute_fn_params(self) -> list[Parameter]:
    if self._fn is None:
        return []
    owner = get_method_owner(self._fn)
    if owner is not None:
        deps = [dep.pobj for dep in owner.param.method_dependencies(self._fn.__name__)]
        return deps
    dinfo = getattr(self._fn, '_dinfo', {})
    args = list(dinfo.get('dependencies', []))
    kwargs = list(dinfo.get('kw', {}).values())
    return args + kwargs