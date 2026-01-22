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
def _ddp_init_helper(self, parameters, expect_sparse_gradient, param_to_name_mapping, static_graph):
    """
        DDP init helper function to manage parameters, grad hooks, logging, and SyncBatchNorm.

        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
    if static_graph is True or self.find_unused_parameters is False:
        bucket_size_limits = [sys.maxsize]
    else:
        bucket_size_limits = [dist._DEFAULT_FIRST_BUCKET_BYTES, self.bucket_bytes_cap]
    bucket_indices, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(parameters, bucket_size_limits, expect_sparse_gradient)
    if self.mixed_precision is not None:
        for i, p in enumerate(parameters):
            p._idx = i
    self.reducer = dist.Reducer(parameters, list(reversed(bucket_indices)), list(reversed(per_bucket_size_limits)), self.process_group, expect_sparse_gradient, self.bucket_bytes_cap, self.find_unused_parameters, self.gradient_as_bucket_view, param_to_name_mapping, dist._DEFAULT_FIRST_BUCKET_BYTES)
    self.logger = dist.Logger(self.reducer)
    self.reducer.set_logger(self.logger)
    has_sync_bn = False
    for submodule in self.module.modules():
        if isinstance(submodule, torch.nn.SyncBatchNorm):
            has_sync_bn = True
            break
    self.logger.set_construction_data_and_log(self.module.__class__.__name__, [] if self.device_ids is None else self.device_ids, -1 if self.output_device is None else self.output_device, self.broadcast_buffers, has_sync_bn, static_graph)
    self._passing_sync_batchnorm_handle(self.module)