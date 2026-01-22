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
class RemoteModuleTest(CommonRemoteModuleTest):

    @dist_utils.dist_init
    def test_bad_module(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        remote_device = f'{dst_worker_name}/cpu'
        args = (1,)
        kwargs = dict(first_kwarg=2)
        with self.assertRaisesRegex(ValueError, 'Expect `module_cls\\(\\*args, \\*\\*kwargs\\)` returns an instance of <class nn.Module>,'):
            RemoteModule(remote_device, BadModule, args, kwargs).forward()
        with self.assertRaisesRegex(ValueError, 'Expect `module_cls\\(\\*args, \\*\\*kwargs\\)` returns an instance of <class nn.Module>,'):
            RemoteModule(remote_device, BadModule, args, kwargs).forward()

    @dist_utils.dist_init
    def test_forward_async(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2, '3')
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret_fut = remote_module.forward_async(*args)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_forward_async_script(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        scripted_remote_module = next(self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]))

        @torch.jit.script
        def run_forward_async(scripted_remote_module: RemoteMyModuleInterface):
            ret_fut = scripted_remote_module.forward_async(torch.ones(1), 2, '3')
            ret = ret_fut.wait()
            return ret
        ret = run_forward_async(scripted_remote_module)
        self.assertEqual(ret, ('3', 2, torch.ones(1)))

    @dist_utils.dist_init
    def test_forward_sync(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2, '3')
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret = remote_module.forward(*args)
            self.assertEqual(ret, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_forward_sync_script(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        scripted_remote_module = next(self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]))

        @torch.jit.script
        def run_forward(scripted_remote_module: MyModuleInterface):
            ret = scripted_remote_module.forward(torch.ones(1), 2, '3')
            return ret
        ret = run_forward(scripted_remote_module)
        self.assertEqual(ret, ('3', 2, torch.ones(1)))

    @dist_utils.dist_init
    def test_forward_with_kwargs(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2)
        kwargs = dict(word='3')
        for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
            ret_fut = remote_module.forward_async(*args, **kwargs)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args + ('3',))))
            ret = remote_module.forward(*args, **kwargs)
            self.assertEqual(ret, tuple(reversed(args + ('3',))))

    @dist_utils.dist_init
    def test_remote_parameters(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
            param_rrefs = remote_module.remote_parameters()
            self.assertEqual(len(param_rrefs), 1)
            self.assertTrue(torch.equal(param_rrefs[0].to_here(), _PARAM_VAL))

    @dist_utils.dist_init
    def test_get_module_rref(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
            rref = remote_module.get_module_rref()
            self.assertEqual(rref, remote_module.module_rref)
            for param in rref.to_here().parameters():
                self.assertTrue(torch.equal(param, _PARAM_VAL))

    @dist_utils.dist_init
    def test_train_eval(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
            remote_module.train()
            ret1 = rpc.rpc_sync(dst_worker_name, get_remote_training_arg, args=(remote_module.get_module_rref(),))
            self.assertEqual(ret1, True)
            remote_module.eval()
            ret2 = rpc.rpc_sync(dst_worker_name, get_remote_training_arg, args=(remote_module.get_module_rref(),))
            self.assertEqual(ret2, False)

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

    @dist_utils.dist_init
    def test_send_remote_module_with_a_new_attribute_not_pickled_over_the_wire(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
            new_attr_name = 'new_attr'
            setattr(remote_module, new_attr_name, 1)
            attrs = rpc.rpc_sync(dst_worker_name, remote_module_attributes, (remote_module,))
            self.assertNotIn(new_attr_name, attrs)

    @dist_utils.dist_init
    def test_remote_module_py_pickle_not_supported(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
            with TemporaryFileName() as fname:
                with self.assertRaisesRegex(RuntimeError, 'Cannot pickle RemoteModule in python pickler. RemoteModule can only be pickled when using RPC'):
                    torch.save(remote_module, fname)

    @dist_utils.dist_init
    def test_remote_module_py_pickle_not_supported_script(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]):
            with TemporaryFileName() as fname:
                with self.assertRaisesRegex(torch.jit.Error, 'can only be pickled when using RPC'):
                    torch.save(remote_module, fname)