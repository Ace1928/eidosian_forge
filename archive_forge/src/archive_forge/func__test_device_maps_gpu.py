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
def _test_device_maps_gpu(self, x_from, y_from, z_to, device_map, dst=None, fn=None):
    fn = TensorPipeAgentCudaRpcTest._gpu_add_given_devices if fn is None else fn
    x_to = device_map[x_from]
    y_to = device_map[y_from]
    options = self.rpc_backend_options
    dst = worker_name((self.rank + 1) % self.world_size) if dst is None else dst
    options.set_device_map(dst, device_map)
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
    x = torch.zeros(2).to(x_from)
    y = torch.ones(2).to(y_from)
    ret = rpc.rpc_sync(dst, fn, args=(x, y, x_to, y_to, z_to))
    reverse_device_map = {device_map[k]: k for k in device_map}
    z_from = reverse_device_map[z_to]
    ret_device = 'cpu' if ret.device.type == 'cpu' else ret.device.index
    self.assertEqual(ret_device, z_from)
    self.assertEqual(ret, torch.ones(2).to(z_from))
    rpc.shutdown()