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
def test_rref_jit_pickle_not_supported(self):
    n = self.rank + 1
    dst_rank = n % self.world_size
    rref_var = rpc_return_rref(worker_name(dst_rank))
    with TemporaryFileName() as fname:
        with self.assertRaisesRegex(RuntimeError, 'RRef jit pickling is only allowed inside RPC calls'):
            save_rref(rref_var, fname)