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
def _backward_multiple_round_trips(self, t1, t2, t3, t4, t5, local_grads, sparse):
    for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
        with dist_autograd.context() as context_id:
            val = self._exec_func(exec_mode, torch.add, t1, t2)
            val = self._exec_func(exec_mode, torch.mul, t3, val)
            s1 = self._exec_func(exec_mode, torch.stack, (t4, val))
            s2 = self._exec_func(exec_mode, torch.stack, (t5, val))
            if sparse:
                val = self._exec_func(exec_mode, torch.mul, s1, s2)
                val = self._exec_func(exec_mode, torch.mul, val, val)
                loss = torch.sparse.sum(val)
            else:
                val = self._exec_func(exec_mode, torch.bmm, s1, s2)
                val = self._exec_func(exec_mode, torch.matmul, val, val)
                loss = val.sum()
            ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t3, t4, t5)
            local_grads = ret if ret else local_grads