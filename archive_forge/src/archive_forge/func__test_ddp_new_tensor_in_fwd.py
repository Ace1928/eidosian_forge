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
def _test_ddp_new_tensor_in_fwd(self, static_graph):

    class MyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 10, bias=False)
            self.fc2 = nn.Linear(10, 10, bias=False)
            self.device = self.fc1.weight.device

        def __init_opt(self):
            opt = torch.randn(1, 10, device=self.device)
            return opt

        def forward(self, x, opt_1, opt_2, opt_nested):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            if opt_1 is None:
                opt_1 = self.__init_opt()
            if opt_2 is None:
                opt_2 = self.__init_opt()
            if opt_nested is None or not torch.is_tensor(opt_nested):
                opt_nested = self.__init_opt()
            return (x, opt_1, opt_2, {'tensor': opt_nested})
    model = MyModel().to(self.rank)
    for find_unused in [True, False]:
        ddp = DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank, broadcast_buffers=False, find_unused_parameters=find_unused, static_graph=static_graph)
        opt = [None for _ in range(3)]
        for i in range(2):
            ddp.zero_grad()
            x = torch.randn(1, 10, device=self.rank)
            out, opt[0], opt[1], opt[2] = ddp(x, opt_1=opt[0], opt_2=opt[1], opt_nested=opt[2])
            for i in range(len(opt)):
                if torch.is_tensor(opt[i]):
                    self.assertEqual(opt[i].grad_fn, None)
                else:
                    self.assertEqual(opt[i]['tensor'].grad_fn, None)
            out.mean().backward()