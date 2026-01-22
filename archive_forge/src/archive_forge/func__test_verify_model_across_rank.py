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
def _test_verify_model_across_rank(self, use_logger):
    group_gloo = dist.new_group(timeout=timedelta(seconds=60), backend=dist.Backend.GLOO)
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    group_to_use = dist.new_group(backend=dist.get_backend(), timeout=timedelta(seconds=5))
    torch.cuda.set_device(self.rank)
    ctx, expected_err = self._determine_expected_error_verify_model_across_rank(group_to_use)
    net = EmbeddingNetDifferentParams(0)
    net = torch.nn.parallel.DistributedDataParallel(net.to(self.rank), device_ids=[self.rank], process_group=group_to_use)
    net.module.lin = nn.Linear(100 if self.rank == 0 else 10, 1)
    with ctx:
        if use_logger:
            _verify_param_shape_across_processes(net.process_group, list(net.parameters()), net.logger)
        else:
            _verify_param_shape_across_processes(net.process_group, list(net.parameters()))
        dist.barrier(group_to_use)
    if use_logger and self.rank != 0:
        verify_ddp_error_logged(net, expected_err)
    dist.barrier(group_gloo)