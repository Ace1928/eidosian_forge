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
def _profiler_test_with_rpc(self, rpc_exec_mode, func, args, use_record_function=False, dst=None, kineto_profile=False):
    dst = dst if dst is not None else (self.rank + 1) % self.world_size
    p = _profile if not kineto_profile else torch.profiler.profile
    if self.rank == 1:
        with p() as prof:
            record_function_ctx_mgr = contextlib.nullcontext() if not use_record_function else torch.autograd.profiler.record_function('foo')
            with record_function_ctx_mgr as rf:
                if rpc_exec_mode == RPCExecMode.SYNC:
                    rpc.rpc_sync(worker_name(dst), func, args=args)
                elif rpc_exec_mode == RPCExecMode.ASYNC:
                    fut = rpc.rpc_async(worker_name(dst), func, args=args)
                    if kineto_profile:
                        fut2 = rpc.rpc_async(worker_name(dst), func, args=args)
                        fut2.wait()
                    fut.wait()
                else:
                    self.assertTrue(rpc_exec_mode == RPCExecMode.REMOTE)
                    rref = rpc.remote(worker_name(dst), func, args=args)
                    rref.to_here()
                    rref._get_profiling_future().wait()
        events = prof.function_events if not kineto_profile else prof.events()
        if kineto_profile:
            with self.assertRaises(IndexError):
                get_function_event(events, rpc_exec_mode.value)
            return
        rpc_event = get_function_event(events, rpc_exec_mode.value)
        self.assertEqual(rpc_event.node_id, self.rank)
        remote_events = {event for event in events if event.node_id == dst} - {rpc_event}
        self.assertGreaterEqual(len(remote_events), 1)
        for remote_event in remote_events:
            self.assertEqual(remote_event.node_id, dst)
        if use_record_function:
            scope_event = get_function_event(events, 'foo')
            self.assertLessEqual(scope_event.time_range.start, rpc_event.time_range.start)
            self.assertGreaterEqual(scope_event.time_range.end, rpc_event.time_range.end)
        self_worker_name = worker_name(self.rank)
        dst_worker_name = worker_name(dst)
        self.check_profiling_info(self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode)
        if use_record_function:
            foo_event_ix = next((i for i, event in enumerate(events) if 'foo' in event.name))
            rpc_event_idx = next((i for i, event in enumerate(events) if rpc_exec_mode.value in event.name))
            self.assertLess(foo_event_ix, rpc_event_idx)