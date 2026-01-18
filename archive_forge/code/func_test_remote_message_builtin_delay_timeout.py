import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_REMOTE_CALL': 2, 'SCRIPT_RREF_FETCH_CALL': 1})
def test_remote_message_builtin_delay_timeout(self):
    func = torch.add
    args = (torch.tensor(1), torch.tensor(1))
    self._test_remote_message_delay_timeout(func, args)