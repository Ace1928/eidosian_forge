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
def _test_trainer_ps(self, create_ref_fn, trainer_fn, sparse):
    if sparse:
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
    else:
        t1 = torch.ones((3, 3), requires_grad=True)
        t2 = torch.zeros((3, 3), requires_grad=True)
    local_ret = torch.add(t1, t2)
    if sparse:
        torch.sparse.sum(local_ret).backward()
    else:
        local_ret.sum().backward()
    rref_t1 = rpc.remote(worker_name(self.rank), create_ref_fn, args=())
    rank_diffs = [1, 2, 3]
    futures = []
    for rank_diff in rank_diffs:
        futures.append(rpc.rpc_async(worker_name((self.rank + rank_diff) % self.world_size), trainer_fn, args=(rref_t1, t2, worker_name(self.rank), rank_diff, sparse)))
    for rank_diff in rank_diffs:
        self._check_rpc_done(rank_diff)
    accumulate_grad_func = None
    for rank_diff in rank_diffs:
        ctx_id = ctx_ids[rank_diff]
        grads = dist_autograd.get_gradients(ctx_id)
        local_t1 = rref_t1.to_here()
        self.assertIn(local_t1, grads)
        self.assertEqual(grads[local_t1], t1.grad)
    _set_rpc_done(None, 0)
    torch.futures.wait_all(futures)