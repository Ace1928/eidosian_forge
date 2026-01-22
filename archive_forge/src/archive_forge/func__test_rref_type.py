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
def _test_rref_type(self, blocking):

    def launched_rpc(events):
        expected_name = f'rpc_{RPCExecMode.ASYNC.value}#_rref_typeof_on_owner'
        return any((e.name.startswith(expected_name) for e in events))
    dst = worker_name((self.rank + 1) % self.world_size)
    rref = rpc.remote(dst, torch.add, args=(torch.ones(2), 1))
    with _profile() as p:
        t = rref._get_type(blocking=blocking)
        if not blocking:
            t = t.wait()
    self.assertTrue(launched_rpc(p.function_events))
    expected_type = type(torch.ones(2))
    self.assertEqual(t, expected_type)
    futs = []

    def verify(fut):
        self.assertEqual(fut.value(), expected_type)
    with _profile() as p:
        for _ in range(10):
            t = rref._get_type(blocking=blocking)
            if not blocking:
                futs.append(t)
                t.add_done_callback(verify)
                t = t.wait()
            self.assertEqual(t, expected_type)
    if not blocking:
        first_fut = futs[0]
        for f in futs[1:]:
            self.assertTrue(f is first_fut)
    self.assertFalse(launched_rpc(p.function_events))
    self.assertEqual(t, type(torch.ones(2)))
    rref = rpc.remote(dst, MyClass, args=(0,))
    rref_type = rref._get_type(blocking=blocking)
    if not blocking:
        rref_type = rref_type.wait()
    self.assertEqual(rref_type, MyClass)