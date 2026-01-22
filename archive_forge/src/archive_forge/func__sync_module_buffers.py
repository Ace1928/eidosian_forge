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
def _sync_module_buffers(self, authoritative_rank):
    if not hasattr(self, 'buffer_hook'):
        self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
    else:
        hook = self.buffer_hook.buffer_comm_hook
        state = self.buffer_hook.buffer_comm_hook_state
        futs = hook(state, self.named_module_buffers)
        if futs is not None:
            self.reducer._install_post_backward_futures(futs)