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
def _test_all_gather_coalesced_helper(self, group, group_id, rank, dtype=torch.float):
    if group_id is not None:
        for test_case_id in range(2, 5):
            input_tensors = [_build_multidim_tensor(tensor_id, tensor_id, rank + tensor_id, dtype=dtype) for tensor_id in range(1, test_case_id)]
            output_tensor_lists = [[_build_multidim_tensor(tensor_id, tensor_id, -1, dtype=dtype) for tensor_id in range(1, test_case_id)] for _ in group]
            expected_tensors = [[_build_multidim_tensor(tensor_id, tensor_id, rank_iter + tensor_id, dtype=dtype) for tensor_id in range(1, test_case_id)] for rank_iter in group]
            assert self._run_all_gather_coalesced_and_verify(output_tensor_lists, input_tensors, expected_tensors, group_id), 'output tensors do not match expected ouputs'
    self._barrier()