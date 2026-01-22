import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
class _CachingTorchDispatchMode(TorchDispatchMode):
    """
    A :class:`TorchDispatchMode` to implement selective activation checkpointing
    that's compatible with torch.compile. Used together with _CachedTorchDispatchMode.
    """

    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage

    def push_into_storage(self, out, func, args, kwargs):
        out_detached = tree_map(_detach, out)
        self.storage[func].append(out_detached)

    def _handle_compile_in_forward_ctx(self, should_not_recompute, func, args, kwargs):
        if func in _ignored_ops:
            return func(*args, **kwargs)
        if should_not_recompute:
            fx_traceback.current_meta['recompute'] = 0
        out = func(*args, **kwargs)
        self.push_into_storage(out, func, args, kwargs)
        return out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        should_not_recompute = self.policy_fn('forward', func, *args, **kwargs)
        if _is_compiling(func, args, kwargs):
            return self._handle_compile_in_forward_ctx(should_not_recompute, func, args, kwargs)
        else:
            if should_not_recompute:
                out = func(*args, **kwargs)
                self.push_into_storage(out, func, args, kwargs)
            else:
                out = func(*args, **kwargs)
            return out