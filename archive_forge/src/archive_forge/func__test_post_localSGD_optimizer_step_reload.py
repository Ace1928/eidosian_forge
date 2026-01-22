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
def _test_post_localSGD_optimizer_step_reload(self, create_averager, chkpt_file):
    learning_rate = 0.03
    net_using_post_localSGD_opt = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(DDP_NET).cuda(), device_ids=[self.rank])
    averager = create_averager()
    post_localSGD_opt = self._create_post_localSGD_optimizer(net_using_post_localSGD_opt, learning_rate, averager)
    averager2 = create_averager()
    dummy_post_localSGD_opt = self._create_post_localSGD_optimizer(net_using_post_localSGD_opt, learning_rate, averager2)
    input = torch.randn(dist.get_world_size() * 2, 2).cuda()
    target = torch.randn(dist.get_world_size() * 2, 4).cuda()
    loss_fn = nn.MSELoss()
    for _ in range(20):
        self._perform_a_train_step(post_localSGD_opt, net_using_post_localSGD_opt, loss_fn, input, target)
    if self.rank == 0:
        torch.save({'optimizer_state_dict': post_localSGD_opt.state_dict()}, chkpt_file)
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
    checkpoint = torch.load(chkpt_file, map_location=map_location)
    dummy_post_localSGD_opt.load_state_dict(checkpoint['optimizer_state_dict'])
    self.assertNotEqual(averager2.step, 0)
    self.assertEqual(averager.step, averager2.step)
    del checkpoint['optimizer_state_dict']['step']
    self.assertNotIn('step', checkpoint['optimizer_state_dict'])
    with self.assertWarnsRegex(expected_warning=UserWarning, expected_regex='Loaded state dict does not contain a step counter for an averager. Setting step counter to 0.'):
        dummy_post_localSGD_opt.load_state_dict(checkpoint['optimizer_state_dict'])
    self.assertEqual(averager2.step, 0)