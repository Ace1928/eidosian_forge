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
class _DDPJoinHook(JoinHook):

    def __init__(self, ddp, divide_by_initial_world_size):
        """Set config variables for internal usage."""
        assert isinstance(ddp, DistributedDataParallel), 'DDP join hook requires passing in a DistributedDataParallel instance as the state'
        assert ddp.logger is not None
        ddp.logger._set_uneven_input_join()
        self.ddp = ddp
        self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()

    def main_hook(self):
        """Shadow the DDP collective communication operations in the forward and backward passes."""
        ddp = self.ddp
        ddp.reducer._rebuild_buckets()
        ddp._check_and_sync_module_buffers()
        work = ddp._check_global_requires_backward_grad_sync(is_joined_rank=True)
        work.wait()
        should_sync_backwards = work.result()[0].item() != 0
        ddp.require_forward_param_sync = should_sync_backwards
        if not should_sync_backwards:
            return
        ddp._match_all_reduce_for_bwd_pass()
        if ddp.find_unused_parameters:
            ddp._match_unused_params_allreduce()
        ddp.reducer._push_all_rebuilt_params()

    def post_hook(self, is_last_joiner: bool):
        """Sync the final model to ensure that the model is the same across all processes."""
        self.ddp._sync_final_model(is_last_joiner)