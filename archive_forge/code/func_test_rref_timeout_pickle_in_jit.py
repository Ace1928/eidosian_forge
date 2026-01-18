from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=['SCRIPT_REMOTE_CALL'])
def test_rref_timeout_pickle_in_jit(self):
    if self.rank != 0:
        return
    dst_rank = (self.rank + 1) % self.world_size
    dst_worker = f'worker{dst_rank}'
    rref = rpc.remote(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
    wait_until_pending_futures_and_users_flushed()
    with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
        rpc_async_with_rref_arg(dst_worker, (rref,))