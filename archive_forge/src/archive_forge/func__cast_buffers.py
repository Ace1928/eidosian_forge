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
def _cast_buffers(mixed_precision_config, root_module):
    """Casts buffers to the given ``buffer_dtype``."""
    for buf in root_module.buffers():
        if hasattr(buf, '_ddp_ignored') and buf._ddp_ignored:
            continue
        buf.data = buf.to(dtype=mixed_precision_config.buffer_dtype)