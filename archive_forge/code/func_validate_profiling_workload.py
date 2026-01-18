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
def validate_profiling_workload(self, dst, prof):

    def convert_remote_to_local(event_name):
        return event_name[event_name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR):]
    events = prof.function_events
    remote_events = {convert_remote_to_local(event.name): event for event in events if event.is_remote}
    self.assertTrue('aten::mul' in remote_events)
    remote_mul_event = remote_events['aten::mul']
    self.assertEqual(remote_mul_event.node_id, dst)
    self.check_profiling_info(worker_name(self.rank), worker_name(dst), torch.mul, remote_mul_event, RPCExecMode.ASYNC)