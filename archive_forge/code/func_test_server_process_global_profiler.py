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
def test_server_process_global_profiler(self):
    if self.rank != 0:
        return
    dst_rank = (self.rank + 1) % self.world_size
    dst_worker_name = worker_name(dst_rank)
    x = torch.tensor(1)
    y = torch.tensor(2)
    outer_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
    outer_profile_rref.rpc_sync().__enter__()
    rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
    inner_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
    inner_profile_rref.rpc_sync().__enter__()
    rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
    inner_profile_rref.rpc_sync().__exit__(None, None, None)
    outer_profile_rref.rpc_sync().__exit__(None, None, None)
    inner_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (inner_profile_rref,))
    expected_inner_events = ['aten::sub']
    expected_outer_events = expected_inner_events + ['aten::add']
    self._assert_top_level_events(inner_events, expected_inner_events)
    outer_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (outer_profile_rref,))
    self._assert_top_level_events(outer_events, expected_outer_events)
    inner_profile_rref.rpc_sync().key_averages()
    outer_profile_rref.rpc_sync().key_averages()