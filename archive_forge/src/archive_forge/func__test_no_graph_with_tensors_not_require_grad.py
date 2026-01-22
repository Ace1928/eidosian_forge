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
def _test_no_graph_with_tensors_not_require_grad(self, exec_mode, sparse):
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    dst_rank = (self.rank + 1) % self.world_size
    with dist_autograd.context() as context_id:
        if sparse:
            t1 = build_sparse_tensor(requires_grad=False)
            t2 = build_sparse_tensor(requires_grad=False)
        else:
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
        if ExecMode.RPC_SYNC == exec_mode:
            ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
        elif ExecMode.REMOTE == exec_mode:
            ret = rpc.remote(worker_name(dst_rank), torch.add, args=(t1, t2)).to_here()
        else:
            raise ValueError(f'Unrecognized ExecMode {exec_mode}')
        rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
        ctx = dist_autograd._current_context()
        send_functions = ctx._send_functions()
        self.assertEqual(len(send_functions), 0)
        recv_functions = ctx._recv_functions()
        self.assertEqual(len(recv_functions), 0)
        self._check_rpc_done(1)
        self.assertNotEqual(-1, dist_autograd._retrieve_context(ctx_ids[1]))
        dist.barrier()