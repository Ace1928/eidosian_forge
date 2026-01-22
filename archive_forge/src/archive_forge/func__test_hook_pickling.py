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
def _test_hook_pickling(self, hook, hook_state):
    torch.manual_seed(0)
    learning_rate = 0.01
    chkpt_file = tempfile.gettempdir() + '/checkpoint.pt'
    rank = self.rank
    input = torch.randn(7, 1, device=rank)
    target = torch.randn(7, 5, device=rank)
    net = torch.nn.Linear(1, 5).to(rank)
    ddp_model = DistributedDataParallel(copy.deepcopy(net), device_ids=[rank])
    dummy_ddp_model = DistributedDataParallel(copy.deepcopy(net), device_ids=[rank])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate)
    ddp_model.register_comm_hook(hook_state, hook)
    ddp_model.train()
    for _ in range(10):
        optimizer.zero_grad()
        out = ddp_model(input)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
    state = {'state_dict': ddp_model.state_dict(), 'comm_hook': hook, 'comm_hook_state': hook_state}
    if rank == 0:
        with self.assertLogs('torch.distributed') as captured:
            torch.save(state, chkpt_file)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), 'NOTE: Process group is not serializable and excluded from a saved state.')
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    with self.assertLogs('torch.distributed') as captured:
        checkpoint = torch.load(chkpt_file, map_location=map_location)
    self.assertEqual(len(captured.records), 1)
    self.assertEqual(captured.records[0].getMessage(), 'NOTE: Process group will be set to a default group (i.e. the world size).                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded.')
    dummy_ddp_model.load_state_dict(checkpoint['state_dict'])
    dummy_hook = checkpoint['comm_hook']
    dummy_hook_state = checkpoint['comm_hook_state']
    dummy_optimizer = torch.optim.SGD(dummy_ddp_model.parameters(), lr=learning_rate)
    self.assertEqual(dummy_hook.__qualname__, hook.__qualname__)
    self.assertEqual(hook_state.__slots__, dummy_hook_state.__slots__)
    for entry in dummy_hook_state.__slots__:
        if entry != 'process_group' and entry != 'rng':
            self.assertEqual(getattr(dummy_hook_state, entry), getattr(hook_state, entry))
    self.assertEqual(dummy_hook_state.process_group, _get_default_group())
    for entry1, entry2 in zip(hook_state.rng.get_state(), dummy_hook_state.rng.get_state()):
        np.testing.assert_array_equal(entry1, entry2)
    dummy_ddp_model.register_comm_hook(dummy_hook_state, dummy_hook)
    dummy_ddp_model.train()
    for _ in range(10):
        optimizer.zero_grad()
        dummy_optimizer.zero_grad()
        out_origin = ddp_model(input)
        out_dummy = dummy_ddp_model(input)
        loss_origin = F.mse_loss(out_origin, target)
        loss_dummy = F.mse_loss(out_dummy, target)
        loss_origin.backward()
        loss_dummy.backward()
        optimizer.step()
        dummy_optimizer.step()
    for orig_param, dummy_param in zip(ddp_model.parameters(), dummy_ddp_model.parameters()):
        self.assertEqual(orig_param.grad, dummy_param.grad)
    dist.barrier()
    if rank == 0:
        os.remove(chkpt_file)