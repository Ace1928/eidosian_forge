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
@mock.patch.object(torch.distributed.rpc.api, '_delete_all_user_and_unforked_owner_rrefs')
def _test_rref_leak(self, _mock_delete_all_user_and_unforked_owner_rrefs, ignore_leak):
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    dist.barrier()
    rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), torch.add, args=(torch.ones(2, 2), 1))
    import torch.distributed.rpc.api as api
    if ignore_leak:
        api._ignore_rref_leak = True
        rpc.shutdown(graceful=True)
    else:
        api._ignore_rref_leak = False
        with self.assertRaisesRegex(RuntimeError, 'Leaking RRef'):
            rpc.shutdown(graceful=True)