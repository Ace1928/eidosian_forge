import time
import io
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.autograd.profiler_legacy import profile as _profile
@dist_init
def test_some_kwargs_are_populated_by_defaults(self):
    if self.rank != 0:
        return
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)
    args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
    kwargs = {'first_kwarg': torch.tensor([2, 2])}
    for script_op in [script_rpc_async_call, script_rpc_sync_call, script_rpc_remote_call]:
        ret = script_op(dst_worker_name, args, kwargs)
        self.assertEqual(ret, torch.tensor([9, 9]))