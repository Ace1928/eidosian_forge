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
@skip_but_pass_in_sandcastle_if(BACKEND != 'nccl', 'Only Nccl supports CUDA reduce_scatter_tensor')
@skip_if_no_gpu
def test_reduce_scatter_tensor_cuda(self):
    group, group_id, rank = self._init_global_test()
    rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
    size = 2
    tensor_out = torch.zeros(size, dtype=torch.int64)
    tensor_in = torch.arange(len(group) * size)
    tensor_out = self._reduce_scatter_tensor_helper(tensor_out, tensor_in, group_id, rank, True, rank_to_GPU)
    expected_tensor = torch.arange(rank * size, (rank + 1) * size) * len(group)
    self.assertEqual(tensor_out, expected_tensor)
    self._barrier()
    tensor_in = torch.reshape(tensor_in, (len(group), size))
    tensor_out = self._reduce_scatter_tensor_helper(tensor_out, tensor_in, group_id, rank, True, rank_to_GPU)
    self.assertEqual(tensor_out, expected_tensor)
    self._barrier()