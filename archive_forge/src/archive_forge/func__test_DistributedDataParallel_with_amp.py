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
def _test_DistributedDataParallel_with_amp(self, grad_is_view=False):
    torch.manual_seed(31415)
    model = copy.deepcopy(DDP_NET).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    scaler = GradScaler()
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], gradient_as_bucket_view=grad_is_view)
    input = torch.randn(dist.get_world_size() * 2, 2).cuda()
    target = torch.randn(dist.get_world_size() * 2, 4).cuda()
    loss_fn = nn.MSELoss()
    for p in ddp_model.parameters():
        self.assertTrue(p is not None)
        self.assertTrue(p.grad is None)
    for idx in range(20):
        optimizer.zero_grad()
        with autocast():
            output = ddp_model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        for p in ddp_model.parameters():
            if p.requires_grad:
                self.assertTrue(p.grad is not None)
                self.assertFalse(p.grad.isnan().any())
                self.assertFalse(p.grad.isinf().any())
        scaler.step(optimizer)
        scaler.update()
        torch.manual_seed(1337 + idx)
        input = input[torch.randperm(dist.get_world_size() * 2)]
    return ddp_model