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
def test_rpc_timeouts(self):
    dst_rank = (self.rank + 1) % self.world_size
    dst_worker = worker_name(dst_rank)
    timeout = 0.1
    expected_error = self.get_timeout_error_regex()
    fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=timeout)
    with self.assertRaisesRegex(RuntimeError, expected_error):
        fut.wait()
    rpc.rpc_async(dst_worker, my_sleep_func, args=(1,)).wait()
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=timeout)
    rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))
    rpc._set_rpc_timeout(0.001)
    fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,))
    with self.assertRaisesRegex(RuntimeError, expected_error):
        fut.wait()
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))
    rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=5).wait()
    rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=5)
    rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=0).wait()
    rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=0)
    rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)