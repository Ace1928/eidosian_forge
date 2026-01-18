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
def test_unsupported_methods(self):
    if self.rank != 0:
        return
    dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
    for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
        with self.assertRaisesRegex(ValueError, 'Method ``register_buffer`` not supported for RemoteModule'):
            remote_module.register_buffer('buffer', torch.ones(5))
        with self.assertRaisesRegex(ValueError, 'Method ``register_parameter`` not supported for RemoteModule'):
            remote_module.register_parameter('param', torch.nn.Parameter(torch.ones(1)))
        with self.assertRaisesRegex(ValueError, 'Method ``add_module`` not supported for RemoteModule'):
            remote_module.add_module('empty', None)
        with self.assertRaisesRegex(ValueError, 'Method ``apply`` not supported for RemoteModule'):
            fn = torch.rand((3, 3), requires_grad=False)
            remote_module.apply(fn)
        with self.assertRaisesRegex(ValueError, 'Method ``cuda`` not supported for RemoteModule'):
            remote_module.cuda()
        with self.assertRaisesRegex(ValueError, 'Method ``cpu`` not supported for RemoteModule'):
            remote_module.cpu()
        with self.assertRaisesRegex(ValueError, 'Method ``type`` not supported for RemoteModule'):
            remote_module.type(torch.FloatTensor)
        with self.assertRaisesRegex(ValueError, 'Method ``float`` not supported for RemoteModule'):
            remote_module.float()
        with self.assertRaisesRegex(ValueError, 'Method ``double`` not supported for RemoteModule'):
            remote_module.double()
        with self.assertRaisesRegex(ValueError, 'Method ``bfloat16`` not supported for RemoteModule'):
            remote_module.bfloat16()
        with self.assertRaisesRegex(ValueError, 'Method ``to`` not supported for RemoteModule'):
            remote_module.to('cpu', dtype=torch.int32)

        def hook(module, grad_input, grad_output):
            pass
        with self.assertRaisesRegex(ValueError, 'Method ``register_backward_hook`` not supported for RemoteModule'):
            remote_module.register_backward_hook(hook)
        with self.assertRaisesRegex(ValueError, 'Method ``register_forward_pre_hook`` not supported for RemoteModule'):
            remote_module.register_forward_pre_hook(hook)
        with self.assertRaisesRegex(ValueError, 'Method ``register_forward_hook`` not supported for RemoteModule'):
            remote_module.register_forward_hook(hook)
        with self.assertRaisesRegex(ValueError, 'Method ``state_dict`` not supported for RemoteModule'):
            remote_module.state_dict()
        with self.assertRaisesRegex(ValueError, 'Method ``load_state_dict`` not supported for RemoteModule'):
            remote_module.load_state_dict({})
        with self.assertRaisesRegex(ValueError, 'Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead.'):
            remote_module.parameters()
        with self.assertRaisesRegex(ValueError, 'Method ``named_parameters`` not supported for RemoteModule'):
            remote_module.named_parameters()
        with self.assertRaisesRegex(ValueError, 'Method ``buffers`` not supported for RemoteModule'):
            remote_module.buffers()
        with self.assertRaisesRegex(ValueError, 'Method ``named_buffers`` not supported for RemoteModule'):
            remote_module.named_buffers()
        with self.assertRaisesRegex(ValueError, 'Method ``children`` not supported for RemoteModule'):
            remote_module.children()
        with self.assertRaisesRegex(ValueError, 'Method ``named_children`` not supported for RemoteModule'):
            remote_module.named_children()
        with self.assertRaisesRegex(ValueError, 'Method ``modules`` not supported for RemoteModule'):
            remote_module.modules()
        with self.assertRaisesRegex(ValueError, 'Method ``named_modules`` not supported for RemoteModule'):
            remote_module.named_modules()
        with self.assertRaisesRegex(ValueError, 'Method ``requires_grad_`` not supported for RemoteModule'):
            remote_module.requires_grad_()
        with self.assertRaisesRegex(ValueError, 'Method ``zero_grad`` not supported for RemoteModule'):
            remote_module.zero_grad()
        with self.assertRaisesRegex(ValueError, 'Method ``share_memory`` not supported for RemoteModule'):
            remote_module.share_memory()
        with self.assertRaisesRegex(ValueError, 'Method ``extra_repr`` not supported for RemoteModule'):
            remote_module.extra_repr()