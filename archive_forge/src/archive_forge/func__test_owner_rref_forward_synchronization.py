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
def _test_owner_rref_forward_synchronization(self, local_device, remote_device):
    if self.rank == 0:
        options = self.rpc_backend_options
        options.set_device_map('w0', {local_device: remote_device})
        rpc.init_rpc('w0', rank=0, world_size=1, rpc_backend_options=options)
        model = rpc.remote('w0', torch.nn.Linear, (2048, 20000)).remote().to(remote_device)
        for _ in range(30):
            data = torch.rand(2048, 2048).to(local_device)
            output = model.rpc_sync().forward(data)
            v0 = rpc.RRef(output).remote().sum().to_here().item()
            v1 = output.sum().item()
            self.assertEqual(v0, v1)
        rpc.shutdown()