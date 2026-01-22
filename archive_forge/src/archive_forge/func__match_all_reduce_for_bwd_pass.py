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
def _match_all_reduce_for_bwd_pass(self):
    comm_work = []
    grad_buckets = self.reducer._get_zeros_like_grad_buckets()
    for grad_bucket in grad_buckets:
        work = self.reducer._run_comm_hook(grad_bucket)
        comm_work.append(work)
    for work in comm_work:
        work.wait()