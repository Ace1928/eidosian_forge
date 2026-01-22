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
def _sync_buffers(self):
    with torch.no_grad():
        if self._join_config.enable:
            authoritative_rank = self._find_common_rank(self._distributed_rank, True)
        else:
            authoritative_rank = 0
        self._assign_modules_buffers()
        self._sync_module_buffers(authoritative_rank)