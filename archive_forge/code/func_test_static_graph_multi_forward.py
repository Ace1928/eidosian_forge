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
@skip_if_lt_x_gpu(2)
@skip_but_pass_in_sandcastle_if(BACKEND != 'nccl' and BACKEND != 'gloo', 'Only Nccl & Gloo backend support DistributedDataParallel')
def test_static_graph_multi_forward(self):

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.lin(x))
    torch.cuda.set_device(self.rank)
    torch.manual_seed(42 << 1337 % (self.rank + 1))
    model = Net().cuda(self.rank)
    local_model = copy.deepcopy(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], static_graph=True)
    inp = torch.ones(2, 10, device='cuda')
    for _ in range(3):
        model.zero_grad()
        local_model.zero_grad()
        a = model(inp)
        b = model(inp)
        loss = a.sum() + b.sum()
        loss.backward()
        if self.rank == 0:
            inp_clone = inp.clone()
            for _ in range(2):
                a = local_model(inp_clone)
                b = local_model(inp_clone)
                loss = a.sum() + b.sum()
                loss.backward()
            ws = dist.get_world_size()
            for p in local_model.parameters():
                p.grad.data = p.grad / dist.get_world_size()
            for p_ddp, p_local in zip(model.parameters(), local_model.parameters()):
                self.assertTrue(torch.allclose(p_ddp.grad, p_local.grad), f'{p_ddp.grad} vs {p_local.grad}')
    dist.barrier()