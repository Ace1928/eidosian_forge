from typing import Dict, Tuple
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import rpc_async
from torch.testing import FileCheck
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init
def test_restore_context_after_swtich_to_jit_thread(self):
    if self.rank != 0:
        return

    @torch.jit.script
    def forward_script(context_id: int, dst_worker_name: str, t1: Tensor, t2: Tensor) -> Tuple[Tensor, Tensor]:
        res1_fut = rpc.rpc_async(dst_worker_name, local_add, (t1, t1))
        res1 = res1_fut.wait()
        loss1 = res1.sum()
        res2_fut = rpc.rpc_async(dst_worker_name, local_add, (t2, t2))
        res2 = res2_fut.wait()
        loss2 = res2.sum()
        return (loss1, loss2)
    with dist_autograd.context() as context_id:
        t1 = torch.ones((2, 3), requires_grad=True)
        t2 = torch.ones((2, 3), requires_grad=True)
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        loss0, loss1 = forward_script(context_id, dst_worker_name, t1, t2)
        dist_autograd.backward(context_id, [loss0, loss1])
        grad0, grad1 = dist_autograd.get_gradients(context_id)
        self.assertEqual(grad0, grad1)