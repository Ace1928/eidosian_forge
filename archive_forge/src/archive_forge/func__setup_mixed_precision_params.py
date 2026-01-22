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
def _setup_mixed_precision_params(mixed_precision_config, root_module):
    """Create and free storage for the mixed precision parameters."""
    for param in root_module.parameters():
        if hasattr(param, '_ddp_ignored') and param._ddp_ignored:
            continue
        if not hasattr(param, '_mp_param'):
            param._mp_param = torch.zeros_like(param, device=param.device, dtype=mixed_precision_config.param_dtype, requires_grad=param.requires_grad)
            _free_storage(param._mp_param)
            param._fp_param = param.data