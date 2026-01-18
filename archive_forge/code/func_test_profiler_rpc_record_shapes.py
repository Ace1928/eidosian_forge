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
def test_profiler_rpc_record_shapes(self):
    if self.rank != 1:
        return
    dst = (self.rank + 1) % self.world_size
    dst_worker = worker_name(dst)
    t1, t2 = (torch.ones(100), torch.ones(100))
    with _profile(record_shapes=True) as prof:
        rpc.rpc_sync(dst_worker, torch.add, args=(t1, t2))
    function_events = prof.function_events
    remote_events = [event for event in function_events if event.is_remote]
    remote_add_event = next((event for event in remote_events if 'aten::add' in event.name))
    remote_add_input_shapes = remote_add_event.input_shapes
    with _profile(record_shapes=True) as prof:
        torch.add(t1, t2)
    local_function_events = prof.function_events
    local_add_event = next((event for event in local_function_events if 'aten::add' in event.name))
    local_add_input_shapes = local_add_event.input_shapes
    self.assertEqual(remote_add_input_shapes, local_add_input_shapes)