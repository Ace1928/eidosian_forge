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
def _backward_rref(self, callee, rref_owner, t1, t2, local_grads, sparse):
    local_ret = torch.add(t1, t2)
    if sparse:
        local_ret = torch.sparse.sum(local_ret)
    else:
        local_ret = local_ret.sum()
    local_ret.backward()
    with dist_autograd.context() as context_id:
        if sparse:
            rref_t1 = rpc.remote(rref_owner, build_sparse_tensor, args=(False, True))
        else:
            rref_t1 = rpc.remote(rref_owner, _torch_ones, args=((3, 3),), kwargs={'requires_grad': True})
        if callee == rref_owner:
            rref = rpc.remote(callee, my_rref_add, args=(rref_t1, t2))
        else:
            rref = rpc.remote(callee, my_nested_rref_add, args=(rref_owner, rref_t1, t2))
        ret = rref.to_here()
        if sparse:
            ret = torch.sparse.sum(ret)
        else:
            ret = ret.sum()
        dist_autograd.backward(context_id, [ret])
        grads = dist_autograd.get_gradients(context_id)
        self.assertIn(t2, grads)
        self.assertEqual(grads[t2], t2.grad)
        self.assertTrue(rpc.rpc_sync(rref_owner, _compare_owner_value, args=(context_id, rref_t1, t1.grad)))