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
def _run_uneven_inputs_test(self, test_case, iteration_mapping, find_unused_params):
    model = test_case.model
    inp = test_case.inp
    rank = self.rank
    sync_interval = test_case.sync_interval
    torch.cuda.set_device(rank)
    dist.barrier()
    net = torch.nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank], bucket_cap_mb=1, find_unused_parameters=find_unused_params)
    if test_case.hook is not None:
        net.register_comm_hook(test_case.state, test_case.hook)
        print(f'registered hook {test_case.hook}')
    num_iters = iteration_mapping[rank]
    num_iters_tensor = torch.tensor([num_iters], device=torch.cuda.current_device())
    dist.all_reduce(num_iters_tensor, op=dist.ReduceOp.MIN)
    min_num_iters = num_iters_tensor.item()
    total_iters = 0
    if test_case.throw_on_early_termination:
        if min_num_iters == num_iters:
            exception_ctx = self.assertRaisesRegex(RuntimeError, f'Rank {self.rank} exhausted all inputs')
        else:
            exception_ctx = self.assertRaisesRegex(RuntimeError, 'Detected at least one rank that exhausted inputs.')
    else:
        exception_ctx = nullcontext()
    with exception_ctx:
        with net.join(throw_on_early_termination=test_case.throw_on_early_termination):
            for i in range(num_iters):
                if i % sync_interval != 0:
                    context = net.no_sync()
                else:
                    context = nullcontext()
                with context:
                    if isinstance(inp, tuple):
                        loss = net(*inp).sum()
                    else:
                        loss = net(inp).sum()
                    loss.backward()
                    self._model_step(net)
                    torch.cuda.synchronize(device=rank)
                total_iters += 1
    if test_case.throw_on_early_termination:
        self.assertEqual(total_iters, min_num_iters)
    else:
        self.assertGreaterEqual(total_iters, min_num_iters)
    torch.cuda.synchronize(device=rank)
    if not test_case.throw_on_early_termination:
        self.assertTrue(net._authoritative_rank)
        final_rank_tensor = torch.tensor([net._authoritative_rank], device=self.rank)
        tensor_list = [torch.zeros_like(final_rank_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, final_rank_tensor)
        max_rank = dist.get_world_size() - 1
        self.assertSetEqual({max_rank}, {tensor.item() for tensor in tensor_list})
        self.validate_net_equivalence(net)
        ddp_logging_data = net._get_ddp_logging_data()
        self.assertTrue(ddp_logging_data.get('join_uneven_inputs'))
        dist.barrier()