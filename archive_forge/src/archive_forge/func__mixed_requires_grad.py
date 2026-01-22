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
def _mixed_requires_grad(self, t1, t2, sparse):
    for exec_mode in [ExecMode.RPC_SYNC, ExecMode.REMOTE]:
        with dist_autograd.context() as context_id:
            ret = self._exec_func(exec_mode, DistAutogradTest._mixed_requires_grad_operaton, t1, t2)
            self.assertEqual(t1 * t2, ret)
            if sparse:
                loss = torch.sparse.sum(ret)
            else:
                loss = ret.sum()
            dist_autograd.backward(context_id, [loss])
            self.assertTrue(t1.requires_grad)
            self.assertFalse(t2.requires_grad)
            grads = dist_autograd.get_gradients(context_id)
            self.assertIn(t1, grads)
            self.assertNotIn(t2, grads)
            self.assertEqual(t2, grads[t1])