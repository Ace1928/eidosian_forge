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
def _find_common_rank(self, input_rank, rank_cond):
    rank_to_use = torch.tensor([input_rank if rank_cond else -1], device=self.device)
    dist.all_reduce(rank_to_use, op=ReduceOp.MAX, group=self.process_group)
    if rank_to_use.item() == -1:
        self._log_and_throw(ValueError, 'BUG! Expected rank_cond to be true for at least one process. This indicates a bug in PyTorch, please report an issue.')
    return rank_to_use.item()