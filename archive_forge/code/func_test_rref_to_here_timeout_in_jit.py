from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_RREF_FETCH_CALL': 1})
def test_rref_to_here_timeout_in_jit(self):
    if self.rank != 0:
        return
    dst_rank = (self.rank + 1) % self.world_size
    dst_worker = f'worker{dst_rank}'
    rref = rpc.remote(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
    expected_error = self.get_timeout_error_regex()
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rref_to_here_with_timeout(rref, 0.01)
    rref_to_here_with_timeout(rref, 100)