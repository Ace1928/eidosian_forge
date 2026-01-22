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
def _test_DDP_niter(self, model_base, model_DDP, input, target, loss, local_bs, rank, batch_size, test_save, offset=None, world_size=0, zero_grad=False, memory_format=None, n_iter=5):
    for idx in range(n_iter):
        self._test_DDP_helper(model_base, input, target, loss, memory_format=memory_format)
        if offset is None:
            offset = rank * local_bs
        self._test_DDP_helper(model_DDP, input[offset:offset + local_bs], target[offset:offset + local_bs], loss, world_size * local_bs / batch_size if world_size != 0 else 1, memory_format=memory_format)
        if zero_grad:
            self._model_step_with_zero_grad(model_base)
            self._model_step_with_zero_grad(model_DDP)
        else:
            self._model_step(model_base)
            self._model_step(model_DDP)
        self._assert_equal_param(list(model_base.parameters()), list(model_DDP.module.parameters()))
        input = input[torch.randperm(batch_size)]
        if test_save and idx == 2 and INIT_METHOD.startswith('file://'):
            with tempfile.NamedTemporaryFile() as tmp:
                if sys.platform == 'win32':
                    torch.save(model_DDP, tmp)
                    tmp.seek(0)
                    model_DDP = torch.load(tmp)
                else:
                    torch.save(model_DDP, tmp.name)
                    model_DDP = torch.load(tmp.name)
    with tempfile.TemporaryFile() as tmp_file:
        torch.save(model_DDP, tmp_file)
        tmp_file.seek(0)
        saved_model = torch.load(tmp_file)
    for k in model_DDP.state_dict():
        self.assertEqual(model_DDP.state_dict()[k], saved_model.state_dict()[k])