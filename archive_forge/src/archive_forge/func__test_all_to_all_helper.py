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
def _test_all_to_all_helper(self, group, group_id, rank, cuda=False, rank_to_GPU=None, dtype=torch.float):
    if group_id is not None:
        size = len(group)
        in_splits = [i + 1 for i in group]
        in_tensors = [torch.ones([in_splits[i], size], dtype=dtype) * rank for i, _ in enumerate(group)]
        out_tensors = [torch.ones([rank + 1, size], dtype=dtype) for _ in group]
        expected_tensors = [torch.ones([rank + 1, size], dtype=dtype) * i for i in group]
        if cuda:
            in_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in in_tensors]
            expected_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in expected_tensors]
            out_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in out_tensors]
        dist.all_to_all(out_tensors, in_tensors, group=group_id)
        for t1, t2 in zip(out_tensors, expected_tensors):
            self.assertEqual(t1, t2)
    self._barrier()