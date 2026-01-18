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
def test_owner_equality(self):
    a = RRef(40)
    b = RRef(50)
    other_rank = (self.rank + 1) % self.world_size
    other_a = rpc.remote(worker_name(other_rank), torch.add, args=(torch.ones(1), 1))
    other_b = rpc.remote(worker_name(other_rank), torch.add, args=(torch.ones(1), 1))
    other_a.to_here()
    other_b.to_here()
    self.assertNotEqual(a.owner(), 23)
    self.assertEqual(other_a.owner(), other_b.owner())
    self.assertNotEqual(a.owner(), other_a.owner())
    self.assertEqual(other_a.owner(), other_a.owner())
    self.assertEqual(other_a.owner(), other_b.owner())
    self.assertEqual(a.owner(), a.owner())
    self.assertEqual(a.owner(), b.owner())
    self.assertEqual(a.owner(), rpc.get_worker_info())
    x = {}
    x[a.owner()] = a
    x[other_a.owner()] = other_a
    self.assertEqual(x[a.owner()], a)
    self.assertEqual(x[b.owner()], a)
    self.assertEqual(x[other_a.owner()], other_a)
    self.assertEqual(x[other_b.owner()], other_a)
    self.assertEqual(len(x), 2)