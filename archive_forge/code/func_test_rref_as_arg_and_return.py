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
def test_rref_as_arg_and_return(self):
    n = self.rank + 1
    dst_rank = n % self.world_size
    local_ret = one_arg(torch.ones(2, 2))
    rref = rpc.remote(worker_name(self.rank), one_arg, args=(torch.ones(2, 2),))
    ret = rpc.rpc_sync(worker_name(dst_rank), rref_to_here, args=(rref,))
    self.assertEqual(ret, local_ret)
    rref1 = rpc.rpc_sync(worker_name(dst_rank), return_rref, args=(rref,))
    self.assertEqual(rref1.to_here(), local_ret)
    rref2 = rpc.remote(worker_name(dst_rank), rref_to_here, args=(rref,))
    self.assertEqual(rref2.to_here(), local_ret)
    rref3 = rpc.remote(worker_name(dst_rank), return_rref, args=(rref,))
    self.assertEqual(rref3.to_here().to_here(), local_ret)