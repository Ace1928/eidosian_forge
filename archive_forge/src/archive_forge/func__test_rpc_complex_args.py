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
def _test_rpc_complex_args(self, exec_mode, sparse):
    with dist_autograd.context() as context_id:
        num_tensors = 10
        tensors = []
        for i in range(num_tensors):
            if sparse:
                tensor = build_sparse_tensor(requires_grad=i % 2 == 0)
            else:
                tensor = torch.ones(3, 3, requires_grad=i % 2 == 0)
            tensors.append(tensor)
        dst_rank = self._next_rank()
        if ExecMode.RPC_SYNC == exec_mode:
            ret = rpc.rpc_sync(worker_name(dst_rank), torch.stack, args=(tensors,))
        elif ExecMode.REMOTE == exec_mode:
            ret = rpc.remote(worker_name(dst_rank), torch.stack, args=(tensors,)).to_here()
        else:
            raise ValueError(f'Unrecognized ExecMode {exec_mode}')
        self.assertEqual(torch.stack(tensors), ret)
        next_funcs = next(iter(dist_autograd._current_context()._send_functions().values())).next_functions
        idx = 0
        for i in range(len(next_funcs)):
            self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[i][0].name())
            self.assertEqual(tensors[i], next_funcs[i][0].variable)
        ctx = dist_autograd._current_context()
        worker_ids = ctx._known_worker_ids()
        self.assertEqual(len(worker_ids), 1)
        self.assertEqual(worker_ids, {dst_rank})