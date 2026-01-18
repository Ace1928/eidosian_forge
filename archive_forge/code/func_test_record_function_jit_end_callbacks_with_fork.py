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
def test_record_function_jit_end_callbacks_with_fork(self):
    sleep_interval = 1
    with _profile() as prof:
        with torch.autograd.profiler.record_function('foo') as rf:
            fut = torch.jit._fork(sleep, sleep_interval)
            rf._call_end_callbacks_on_future(fut)
        fut.wait()
    function_events = prof.function_events
    sleep_event = get_function_event(function_events, 'foo')
    self.assertEqual(sleep_event.name, 'foo')
    self.assertGreaterAlmostEqual(sleep_event.cpu_time * 1e-06, sleep_interval)