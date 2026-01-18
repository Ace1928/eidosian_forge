from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_CALL': 1.5})
def test_timeout_in_torchscript_function(self):
    if self.rank != 0:
        return
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)
    args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
    kwargs = {'first_kwarg': torch.tensor([2, 2]), 'second_kwarg': torch.tensor([3, 3])}
    expected_error = self.get_timeout_error_regex()
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0.5)
    rpc._set_rpc_timeout(0.001)
    with self.assertRaisesRegex(RuntimeError, expected_error):
        script_rpc_async_call(dst_worker_name, args, kwargs)
    ret = rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0)
    self.assertEqual(ret, torch.tensor([8, 8]))
    rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)