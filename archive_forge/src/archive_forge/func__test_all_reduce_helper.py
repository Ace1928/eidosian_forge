import copy
import itertools
import math
import os
import random
import sys
import tempfile
import time
from collections import namedtuple, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import Union, NamedTuple, Callable, Any
import unittest
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils
import torch.nn as nn
import torch.nn.functional as F
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.algorithms.ddp_comm_hooks import (
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.distributed_c10d import (
from torch.distributed.utils import (
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer
from torch.utils.data.distributed import DistributedSampler
def _test_all_reduce_helper(self, group, group_id, rank, op, master_value, worker_value, expected_value, cuda=False, rank_to_GPU=None, dtype=torch.float, async_op=False):
    for src in group:
        curr_value = master_value if rank == src else worker_value
        tensor = _build_tensor(src + 1, dtype=dtype).fill_(curr_value)
        if cuda:
            tensor = tensor.cuda(rank_to_GPU[rank][0])
        if tensor.dtype == torch.complex64:
            tensor_shapes = [torch.view_as_real(tensor).shape]
        else:
            tensor_shapes = [tensor.shape]
        self.call_dist_op(':all_reduce', async_op, dist.all_reduce, tensor, op, group_id, async_op=async_op, tensor_shapes=tensor_shapes)
        if src == 0 and cuda and (dist.get_backend() in CUDA_PROFILING_SUPPORTED_BACKENDS):
            self.call_dist_op(':all_reduce', async_op, dist.all_reduce, tensor, op, group_id, async_op=async_op, profile_cuda=True, tensor_shapes=tensor_shapes)
    self._barrier()