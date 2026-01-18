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
def test_return_local_script_class_rref_in_py_and_use_in_script(self):
    if self.rank != 0:
        return
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)
    rref = rpc.rpc_sync(dst_worker_name, owner_create_rref_my_script_class, args=(self.rank,))

    def use_rref_on_owner(rref: RRef[MyScriptClass]) -> int:
        args = (rref,)
        kwargs: Dict[str, Any] = {}
        fut = rpc.rpc_async(rref.owner(), script_rref_get_value_my_script_class, args, kwargs)
        ret = fut.wait()
        return ret
    ret = use_rref_on_owner(rref)
    self.assertEqual(ret, self.rank)
    use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
    ret = use_rref_on_owner_script(rref)
    self.assertEqual(ret, self.rank)