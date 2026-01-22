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
def _test_rref_forward_synchronization(self, local_device, remote_device):
    options = self.rpc_backend_options
    input_src = worker_name(0)
    model_dst = worker_name(1)
    out_relay = worker_name(2)
    if self.rank == 0:
        options.set_device_map(model_dst, {local_device: remote_device})
        options.set_device_map(out_relay, {local_device: local_device})
    elif self.rank == 1:
        options.set_device_map(input_src, {remote_device: local_device})
    elif self.rank == 2:
        options.set_device_map(model_dst, {local_device: remote_device})
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
    if self.rank == 0:
        rref = rpc.remote(model_dst, MyConvNetForMNIST, args=(remote_device,))
        for _ in range(10):
            rref_input = RRef(torch.randn(200, 1, 28, 28).to(local_device))
            rref_out = rref.remote().forward(rref_input, True)
            out = rpc.remote(out_relay, TensorPipeAgentCudaRpcTest._rref_relay, args=(rref_out,)).to_here()
            expected = rref.rpc_sync().forward(rref_input, True)
            self.assertEqual(out, expected)
    rpc.shutdown()