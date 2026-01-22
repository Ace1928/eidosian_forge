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
def _setup_invalidations(self, depth: int=0):
    """
        Since the parameters of the pipeline can change at any time
        we have to invalidate the internal state of the pipeline.
        To handle both invalidations of the inputs of the pipeline
        and the pipeline itself we set up watchers on both.

        1. The first invalidation we have to set up is to re-evaluate
           the function that feeds the pipeline. Only the root node of
           a pipeline has to perform this invalidation because all
           leaf nodes inherit the same shared_obj. This avoids
           evaluating the same function for every branch of the pipeline.
        2. The second invalidation is for the pipeline itself, i.e.
           if any parameter changes we have to notify the pipeline that
           it has to re-evaluate the pipeline. This is done by marking
           the pipeline as `_dirty`. The next time the `_current` value
           is requested the value is resolved by re-executing the
           pipeline.
        """
    if self._fn is not None:
        for _, params in full_groupby(self._fn_params, lambda x: id(x.owner)):
            fps = [p.name for p in params if p in self._root._fn_params]
            if fps:
                params[0].owner.param._watch(self._invalidate_obj, fps, precedence=-1)
    for _, params in full_groupby(self._internal_params, lambda x: id(x.owner)):
        params[0].owner.param._watch(self._invalidate_current, [p.name for p in params], precedence=-1)