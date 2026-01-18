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
def test_send_to_rank(self):
    dst_rank = (self.rank + 1) % self.world_size
    for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
        ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)
    for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
        with self.assertRaises(RuntimeError):
            self._run_func_in_mode(self.world_size + 1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
    for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
        with self.assertRaises(RuntimeError):
            self._run_func_in_mode(-1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
    for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
        with self.assertRaises(ValueError):
            self._run_func_in_mode(dst_rank + 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
    for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
        with self.assertRaises(ValueError):
            self._run_func_in_mode(dst_rank - 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))