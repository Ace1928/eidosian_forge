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
def _check_default_group(self):
    pickle_not_supported = False
    try:
        if self.process_group != _get_default_group():
            pickle_not_supported = True
    except RuntimeError:
        pickle_not_supported = True
    if pickle_not_supported:
        self._log_and_throw(RuntimeError, 'DDP Pickling/Unpickling are only supported when using DDP with the default process group. That is, when you have called init_process_group and have not passed process_group argument to DDP constructor')