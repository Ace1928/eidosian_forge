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
def test_rref_is_owner(self):
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)
    rref_var = rpc_return_rref(dst_worker_name)

    @torch.jit.script
    def rref_tensor_is_owner(rref_var: RRef[Tensor]) -> bool:
        return rref_var.is_owner()
    res = rref_tensor_is_owner(rref_var)
    self.assertEqual(res, False)