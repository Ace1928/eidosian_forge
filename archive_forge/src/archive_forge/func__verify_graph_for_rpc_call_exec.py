import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
def _verify_graph_for_rpc_call_exec(self, send_function):
    next_funcs = send_function.next_functions
    self.assertEqual(1, len(next_funcs))
    add_backward_fn = next_funcs[0][0]
    self.assertEqual('AddBackward0', add_backward_fn.name())
    next_funcs = add_backward_fn.next_functions
    self.assertEqual(2, len(next_funcs))
    self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())
    self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[1][0].name())
    self.assertEqual(next_funcs[0][0], next_funcs[1][0])