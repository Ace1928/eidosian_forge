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
def _verify_backwards_remote(self, tensors, context_id, local_grads, *args):
    dist_autograd.backward(context_id, tensors)
    grads = dist_autograd.get_gradients(context_id)
    nargs = len(args)
    ngrads = 0
    for i in range(0, nargs):
        if local_grads[i] is not None:
            self.assertIn(args[i], grads)
            self.assertEqual(local_grads[i], grads[args[i]])
            ngrads += 1
        else:
            self.assertNotIn(args[i], grads)
    self.assertEqual(ngrads, len(grads))