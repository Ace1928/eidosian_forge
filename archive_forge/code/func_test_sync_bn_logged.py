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
@skip_if_lt_x_gpu(2)
@skip_but_pass_in_sandcastle_if(BACKEND != 'nccl' and BACKEND != 'gloo', 'Only Nccl & Gloo backend support DistributedDataParallel')
def test_sync_bn_logged(self):
    model = BN_NET
    rank = self.rank
    model_gpu = model.cuda(rank)
    no_sync_bn = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(model_gpu), device_ids=[self.rank])
    ddp_logging_data = no_sync_bn._get_ddp_logging_data()
    sync_bn_logged = ddp_logging_data.get('has_sync_bn', True)
    self.assertFalse(sync_bn_logged)
    model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(model_gpu)
    model_DDP = torch.nn.parallel.DistributedDataParallel(model_DDP, device_ids=[self.rank])
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    sync_bn_logged = ddp_logging_data.get('has_sync_bn', False)
    self.assertTrue(sync_bn_logged)