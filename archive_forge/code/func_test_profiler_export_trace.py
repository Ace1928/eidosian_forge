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
@dist_init
def test_profiler_export_trace(self):
    if self.rank != 1:
        return
    dst = (self.rank + 1) % self.world_size
    dst_worker = worker_name(dst)
    with _profile() as p:
        fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
        res = fut.wait()
    events = p.function_events
    with TemporaryFileName() as fname:
        path = fname
        p.export_chrome_trace(path)
        with open(path) as f:
            trace = json.load(f)
            event_names = [event['name'] for event in trace]
            for expected_event_name in EXPECTED_REMOTE_EVENTS + [RPCExecMode.ASYNC.value]:
                event_exists = any((expected_event_name in event_name for event_name in event_names))
                self.assertTrue(event_exists)