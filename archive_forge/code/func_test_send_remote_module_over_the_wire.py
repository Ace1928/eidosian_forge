import enum
from typing import Tuple
import torch
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch import Tensor, nn
from torch._jit_internal import Future
from torch.distributed.nn import RemoteModule
from torch.distributed.nn.api.remote_module import _REMOTE_MODULE_PICKLED_ATTRIBUTES
from torch.distributed.nn.api.remote_module import _RemoteModule
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_utils.dist_init
def test_send_remote_module_over_the_wire(self):
    if self.rank != 0:
        return
    dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
    dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)
    expected_unpickled_attrs = list(_REMOTE_MODULE_PICKLED_ATTRIBUTES)
    expected_unpickled_attrs.append('forward_async')
    expected_unpickled_attrs.append('forward')
    for remote_module in self._create_remote_module_iter(dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR]):
        attrs = rpc.rpc_sync(dst_worker2_name, remote_module_attributes, (remote_module,))
        self.assertListEqual(list(attrs.keys()), expected_unpickled_attrs)
        self.assertEqual(attrs['on'], 'worker1')
        self.assertEqual(attrs['device'], 'cpu')
        self.assertFalse(attrs['is_device_map_set'])
        self.assertFalse(attrs['is_scriptable'])
        args = (torch.ones(1), 2, '3')
        ret1 = rpc.rpc_sync(dst_worker2_name, remote_forward, (remote_module, args))
        self.assertEqual(ret1, tuple(reversed(args)))
        ret2 = rpc.rpc_sync(dst_worker2_name, remote_forward_async, (remote_module, args))
        self.assertEqual(ret2, tuple(reversed(args)))