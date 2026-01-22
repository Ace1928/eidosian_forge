import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Type
import torch
import torch.distributed as dist
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch._utils import _get_device_index
from ..modules import Module
from .scatter_gather import gather, scatter_kwargs  # noqa: F401
def _module_wait_for_copy_hook(self, module, *args: Any, **kwargs: Any) -> None:
    """Before carrying out computation, wait on the appropriate event to ensure low precision copies have finished."""
    try:
        event = self._submodule_to_event[module].popleft()
    except IndexError:
        return
    event.wait(stream=torch.cuda.current_stream())
    for p in module.parameters(recurse=False):
        if not p.requires_grad or (hasattr(p, '_ddp_ignored') and p._ddp_ignored):
            continue
        tmp = p.expand_as(p)
        grad_acc = tmp.grad_fn.next_functions[0][0]
        hook = grad_acc.register_hook(functools.partial(self._fire_reducer_autograd_hook, p._idx))
        p._ddp_mp_hook_state = (grad_acc, hook)