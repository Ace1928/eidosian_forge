import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event
from threading import Lock
from unittest import mock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
from torch.futures import Future
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_utils import TemporaryFileName
from torch.autograd.profiler_legacy import profile as _profile
@dist_init(setup_rpc=False)
def test_pg_init_no_rpc_init(self):
    dist.init_process_group(backend='gloo', init_method=self.file_init_method, rank=self.rank, world_size=self.world_size)

    class MyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 4)

        def forward(self, x):
            return self.lin(x)
    model = MyModel()
    model.train()
    model = torch.nn.parallel.DistributedDataParallel(model)
    with self.assertRaisesRegex(RuntimeError, 'Current RPC agent is not set! Did you initialize the RPC framework'):
        params = []
        for param in model.parameters():
            params.append(RRef(param))