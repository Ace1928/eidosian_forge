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
def test_wait_all_exit_early_python(self):
    RpcTest.timed_out_rpc_event = Event()
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    dist.barrier()
    dst = worker_name((self.rank + 1) % self.world_size)
    fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
    fut2 = rpc.rpc_async(dst, raise_func)
    fut3 = rpc.rpc_async(dst, raise_func)
    with self.assertRaisesRegex(ValueError, expected_err):
        torch.futures.wait_all([fut1, fut2, fut3])
    RpcTest.timed_out_rpc_event.set()