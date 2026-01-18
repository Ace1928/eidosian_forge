import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init
def test_verify_backend_options(self):
    self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_TENSORPIPE)
    self.assertEqual(self.rpc_backend_options.num_worker_threads, 8)
    self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
    self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)
    self.assertEqual(len(self.rpc_backend_options.messages_to_delay), 2)
    self.assertEqual(self.rpc_backend_options.rpc_timeout, rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)