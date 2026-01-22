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
def _compute_params(self) -> list[Parameter]:
    ps = list(self._fn_params)
    if self._trigger:
        ps.append(self._trigger.param.value)
    prev = self._prev
    while prev is not None:
        for p in prev._params:
            if p not in ps:
                ps.append(p)
        prev = prev._prev
    if self._operation is None:
        return ps
    for ref in resolve_ref(self._operation['fn']):
        if ref not in ps:
            ps.append(ref)
    for arg in list(self._operation['args']) + list(self._operation['kwargs'].values()):
        for ref in resolve_ref(arg):
            if ref not in ps:
                ps.append(ref)
    return ps