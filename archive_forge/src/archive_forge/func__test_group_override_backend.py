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
def _test_group_override_backend(self, initializer):
    if BACKEND == 'gloo':
        new_backend = 'nccl'
    elif BACKEND == 'nccl':
        new_backend = 'gloo'
    elif BACKEND in DistTestCases.backend_feature['plugin']:
        new_backend = 'gloo'
    group, group_id, rank = initializer(backend=new_backend)
    if group_id is None:
        return
    if new_backend == 'gloo':
        self.assertTrue(group_id._get_backend_name(), 'gloo')
    if new_backend == 'nccl':
        self.assertTrue(group_id._get_backend_name(), 'nccl')
    self.assertEqual(rank, group[dist.get_rank(group_id)])
    self.assertEqual(len(group), dist.get_world_size(group_id))
    group_rank = dist.get_rank(group_id)
    torch.cuda.set_device(group_rank)
    tensor = _build_tensor(2, value=group_rank).cuda()
    dist.broadcast(tensor, src=group[0], group=group_id)
    self.assertEqual(_build_tensor(2, value=0), tensor.to('cpu'))