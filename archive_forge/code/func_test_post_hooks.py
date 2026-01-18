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
@dist_init
def test_post_hooks(self):
    self.hook_called_times = 0

    def post_hook_add_one(output_grads, input_grads):
        self.hook_called_times += 1
        return output_grads

    def post_hook_add_two(output_grads, input_grads):
        self.hook_called_times += 2
        return output_grads
    t = torch.rand(10, 10, requires_grad=True)
    a = t + t
    accumulate_grad_0 = a.grad_fn.next_functions[0][0]
    accumulate_grad_0.register_hook(post_hook_add_one)
    accumulate_grad_0.register_hook(post_hook_add_two)
    accumulate_grad_1 = a.grad_fn.next_functions[1][0]
    accumulate_grad_1.register_hook(post_hook_add_two)
    with dist_autograd.context() as context_id:
        loss = a.sum()
        dist_autograd.backward(context_id, [loss])
        self.assertEqual(5, self.hook_called_times)
        grads = dist_autograd.get_gradients(context_id)
        self.assertEqual(1, len(grads))
        self.assertTrue(t in grads)