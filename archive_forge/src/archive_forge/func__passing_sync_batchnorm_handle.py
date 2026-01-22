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
def _passing_sync_batchnorm_handle(self, module):
    for layer in module.modules():
        if isinstance(layer, torch.nn.modules.SyncBatchNorm):
            if self.device_type == 'cpu':
                self._log_and_throw(ValueError, 'SyncBatchNorm layers only work with GPU modules')