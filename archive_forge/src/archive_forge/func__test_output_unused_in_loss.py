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
def _test_output_unused_in_loss(self, module_cls, gradient_as_bucket_view):
    model = module_cls()
    local_net = copy.deepcopy(model)
    net = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(model).cuda(self.rank), device_ids=[self.rank], find_unused_parameters=True)
    inp = torch.randn(10, 10)
    if module_cls == DictOutputModule:
        a, b = local_net(inp)['predictions']
        a_dist, b_dist = net(inp)['predictions']
    else:
        a, b = local_net(inp)
        a_dist, b_dist = net(inp)
    loss_dist = b_dist.sum()
    loss_dist.backward()
    if module_cls == DictOutputModule:
        self.assertTrue(net.module.module.a.weight.grad is None)
        self.assertEqual(net.module.module.a.weight.grad, local_net.module.a.weight.grad)
    else:
        self.assertTrue(net.module.a.weight.grad is None)
        self.assertEqual(net.module.a.weight.grad, local_net.a.weight.grad)
    saved_a_local_grad = None
    saved_a_dist_grad = None
    net.zero_grad()
    local_net.zero_grad()
    for i in range(6):
        if module_cls == DictOutputModule:
            a, b = local_net(inp)['predictions']
            a_dist, b_dist = net(inp)['predictions']
        else:
            a, b = local_net(inp)
            a_dist, b_dist = net(inp)
        if i < 2:
            t = a @ b
            t_dist = a_dist @ b_dist
            loss = t.sum()
            loss_dist = t_dist.sum()
        else:
            loss = b.sum()
            loss_dist = b_dist.sum()
        loss.backward()
        loss_dist.backward()
        if i == 1:
            if module_cls == DictOutputModule:
                saved_a_local_grad = local_net.module.a.weight.grad
                saved_a_dist_grad = net.module.module.a.weight.grad
            else:
                saved_a_local_grad = local_net.a.weight.grad
                saved_a_dist_grad = net.module.a.weight.grad
            self.assertEqual(saved_a_local_grad, saved_a_dist_grad)
        elif i >= 2:
            if module_cls == DictOutputModule:
                self.assertEqual(net.module.module.a.weight.grad, saved_a_dist_grad)
                self.assertEqual(local_net.module.a.weight.grad, saved_a_local_grad)
            else:
                self.assertEqual(net.module.a.weight.grad, saved_a_dist_grad)
                self.assertEqual(local_net.a.weight.grad, saved_a_local_grad)
        for local_param, dist_param in zip(local_net.parameters(), net.parameters()):
            local_grad = local_param.grad
            dist_grad = dist_param.grad
            self.assertEqual(local_grad, dist_grad)
    dist.barrier()