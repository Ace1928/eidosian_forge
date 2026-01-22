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
class TestDistBackend(MultiProcessTestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['MASTER_ADDR'] = str(MASTER_ADDR)
        super().setUpClass()

    def setUp(self):
        super().setUp()
        initialize_temp_directories()
        Barrier.init()
        self.skip_return_code_checks = [self.test_ddp_has_finalized.__wrapped__]

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    @property
    def init_method(self):
        return f'{FILE_SCHEMA}{self.file_name}'

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        if BACKEND == 'nccl' and (not torch.cuda.is_available()):
            sys.exit(TEST_SKIPS['no_cuda'].exit_code)
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        if torch.cuda.is_available() and torch.cuda.device_count() < int(self.world_size):
            sys.exit(TEST_SKIPS[f'multi-gpu-{self.world_size}'].exit_code)
        try:
            pg_timeout_seconds = CUSTOM_PG_TIMEOUT.get(test_name, default_pg_timeout)
            timeout = timedelta(seconds=pg_timeout_seconds)
            dist.init_process_group(init_method=self.init_method, backend=BACKEND, world_size=int(self.world_size), rank=self.rank, timeout=timeout)
        except RuntimeError as e:
            if 'recompile' in e.args[0]:
                sys.exit(TEST_SKIPS['backend_unavailable'].exit_code)
            raise
        self._barrier()
        self.run_test(test_name, pipe)
        self._barrier()
        dist.destroy_process_group()
        sys.exit(0)

    @property
    def world_size(self):
        return os.environ['WORLD_SIZE']