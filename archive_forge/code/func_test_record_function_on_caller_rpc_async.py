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
def test_record_function_on_caller_rpc_async(self):
    if self.rank == 0:
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = worker_name(dst_rank)
        block_scope = 'foo'
        with _profile() as prof:
            record_function_on_caller_rpc_async(dst_worker_name, block_scope)
        function_events = prof.function_events
        record_function_scope_event = [event for event in function_events if event.name == block_scope]
        self.assertEqual(1, len(record_function_scope_event))
        record_function_scope_event = record_function_scope_event[0]
        expected_key = _build_rpc_profiling_key(RPCExecMode.ASYNC_JIT, torch._jit_internal._qualified_name(script_add_ones), worker_name(self.rank), dst_worker_name)
        jit_rpc_events = [event for event in function_events if event.name == expected_key]
        self.assertEqual(2, len(jit_rpc_events))
        for jit_rpc_event in jit_rpc_events:
            self.assertTrue(record_function_scope_event.cpu_time_total > jit_rpc_event.cpu_time_total)