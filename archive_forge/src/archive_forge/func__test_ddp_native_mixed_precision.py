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
def _test_ddp_native_mixed_precision(self, gradient_as_bucket_view, set_grad_to_none):
    rank = self.rank
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.set_device(rank)
    inp = torch.randn(10, 1)
    mp_config = self._get_fp16_config()

    class MyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.m = torch.nn.Linear(1, 5)
            self.register_buffer('buffer', torch.randn(1, 2))
            self.p = torch.nn.Parameter(torch.randn(10, 5), requires_grad=False)

        def forward(self_, x):
            params = self_.m.parameters()
            for p in params:
                self.assertEqual(mp_config.param_dtype, p.dtype)
            self.assertEqual(self_.buffer.dtype, mp_config.buffer_dtype)
            self.assertEqual(mp_config.param_dtype, x.dtype)
            return self_.m(x) + self_.p
    m = MyModel()
    net = torch.nn.parallel.DistributedDataParallel(m.to(rank), device_ids=[rank], mixed_precision=mp_config, gradient_as_bucket_view=gradient_as_bucket_view)
    self.assertEqual(net.module.buffer.dtype, mp_config.buffer_dtype)
    for p in net.parameters():
        self.assertEqual(mp_config.param_dtype, p._mp_param.dtype)
        self.assertEqual(torch.float32, p._fp_param.dtype)
    for i in range(6):
        loss = net(inp).sum()
        loss.backward()
        for n, param in net.named_parameters():
            self.assertEqual(param.dtype, torch.float32)
            if param.grad is None:
                assert n == 'module.p'
            else:
                self.assertEqual(param.grad.dtype, torch.float32)
                tensor_list = [torch.zeros_like(param.grad) for _ in range(dist.get_world_size(net.process_group))]
                dist.all_gather(tensor_list, param.grad)
                g, rest = (tensor_list[0], tensor_list[1:])
                self.assertEqual(g.dtype, torch.float32)
                for g_ in rest:
                    self.assertEqual(g_.dtype, torch.float32)
                    self.assertEqual(g, g_)
        net.zero_grad(set_to_none=set_grad_to_none)