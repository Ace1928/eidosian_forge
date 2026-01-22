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
def _test_rref_proxy_class(self, dst):
    rref = rpc.remote(dst, MyClass, args=(7,))
    expected = MyClass(7)
    self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
    self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
    self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())
    expected.increment_value(3)
    self.assertEqual(None, rref.rpc_sync().increment_value(1))
    self.assertEqual(None, rref.rpc_async().increment_value(1).wait())
    self.assertEqual(None, rref.remote().increment_value(1).to_here())
    self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
    self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
    self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())
    self.assertEqual(expected.my_instance_method(2), rref.rpc_sync().my_instance_method(2))
    self.assertEqual(expected.my_instance_method(3), rref.rpc_async().my_instance_method(3).wait())
    self.assertEqual(expected.my_instance_method(4), rref.remote().my_instance_method(4).to_here())
    self.assertEqual(expected.my_static_method(9), rref.rpc_sync().my_static_method(9))
    self.assertEqual(expected.my_static_method(10), rref.rpc_async().my_static_method(10).wait())
    self.assertEqual(expected.my_static_method(11), rref.remote().my_static_method(11).to_here())
    self.assertEqual(expected.my_class_method(2, torch.zeros(2, 2)), rref.rpc_sync().my_class_method(2, torch.zeros(2, 2)))
    self.assertEqual(expected.my_class_method(2, torch.ones(3, 3)), rref.rpc_async().my_class_method(2, torch.ones(3, 3)).wait())
    self.assertEqual(expected.my_class_method(2, torch.ones(4, 4)), rref.remote().my_class_method(2, torch.ones(4, 4)).to_here())