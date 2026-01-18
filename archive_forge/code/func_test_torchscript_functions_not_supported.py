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
def test_torchscript_functions_not_supported(self):
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)
    my_local_script_module = MyScriptModule(self.rank)
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    dist.barrier()
    ret = rpc.rpc_sync(dst_worker_name, MyScriptClass, args=(self.rank,))
    with self.assertRaisesRegex(TypeError, 'pickle'):
        ret = rpc.rpc_async(dst_worker_name, my_local_script_module.forward, args=())