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
def _test_all_to_all_single_equal_split_helper(self, group, group_id, rank, cuda=False, rank_to_GPU=None, dtype=torch.float):
    if group_id is not None:
        size = len(group)
        in_tensor = torch.ones([size, size], dtype=dtype) * rank
        expected_tensor = torch.cat([torch.ones([1, size], dtype=dtype) * i for i in group])
        out_tensor = torch.ones([size, size], dtype=dtype) * -1
        if cuda:
            in_tensor = in_tensor.cuda(rank_to_GPU[rank][0])
            expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
            out_tensor = out_tensor.cuda(rank_to_GPU[rank][0])
        if dtype == torch.complex64:
            tensor_shapes = [torch.view_as_real(in_tensor).shape]
        else:
            tensor_shapes = [in_tensor.shape]
        self.call_dist_op(':all_to_all', False, dist.all_to_all_single, out_tensor, in_tensor, group=group_id, tensor_shapes=tensor_shapes)
        self.assertEqual(out_tensor, expected_tensor)
    self._barrier()