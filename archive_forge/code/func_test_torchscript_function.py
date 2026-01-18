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
def test_torchscript_function(self):
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)
    local_ret = one_arg(torch.ones(2, 2))
    ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
    self.assertEqual(ret, local_ret)
    rref = rpc.remote(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
    self.assertEqual(rref.to_here(), local_ret)
    local_rref = rpc.remote(worker_name(self.rank), one_arg, args=(torch.ones(2, 2),))
    self.assertEqual(local_rref.to_here(), local_ret)