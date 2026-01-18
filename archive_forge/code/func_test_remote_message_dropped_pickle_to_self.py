import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=['PYTHON_REMOTE_CALL'])
def test_remote_message_dropped_pickle_to_self(self):
    self._test_remote_message_dropped_pickle(self.rank)