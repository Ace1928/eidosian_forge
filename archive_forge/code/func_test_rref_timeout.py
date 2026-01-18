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
def test_rref_timeout(self):
    if self.rank != 0:
        return
    dst_rank = (self.rank + 1) % self.world_size
    dst_worker = f'worker{dst_rank}'
    rref = rpc.remote(dst_worker, my_sleep_func, args=(2,), timeout=0.01)
    expected_error = self.get_timeout_error_regex()
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rref._get_future().wait()
    wait_until_pending_futures_and_users_flushed()
    with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
        rref.to_here()
    wait_until_owners_and_forks_on_rank(1, 1, rank=1)