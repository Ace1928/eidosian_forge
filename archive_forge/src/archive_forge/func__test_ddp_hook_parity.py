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
def _test_ddp_hook_parity(self, state, hook, num_validated_iters=100):
    rank = self.rank
    m = torch.nn.Linear(1, 5)
    try:
        process_group = state.process_group
    except AttributeError:
        process_group = state
    net_with_hook = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(m).to(rank), device_ids=[rank], process_group=process_group)
    net_with_hook.register_comm_hook(state=state, hook=hook)
    net_without_hook = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(m).to(rank), device_ids=[rank], process_group=process_group)
    for i in range(100):
        for g in [net_without_hook.module.weight.grad, net_with_hook.module.weight.grad]:
            if g is not None:
                g.requires_grad_(False)
                g.zero_()
        batch = torch.tensor([rank]).float().cuda(rank)
        loss = net_without_hook(batch).sum()
        loss.backward()
        grad = net_without_hook.module.weight.grad
        avg = grad.clone()
        expected_grad = sum((i for i in range(dist.get_world_size()))) / dist.get_world_size()
        loss_hook = net_with_hook(batch).sum()
        loss_hook.backward()
        grad_hook = net_with_hook.module.weight.grad
        avg_hook = grad_hook.clone()
        if i < num_validated_iters:
            self.assertEqual(avg_hook[0, 0].item(), expected_grad, msg=f'Expected hook grad of {expected_grad} but got {avg_hook[0, 0]}')
            self.assertEqual(avg_hook[0, 0], avg[0, 0], msg=f'Expected hook grad to be close to allreduce {avg[0, 0]}, but got {avg_hook[0, 0]}')