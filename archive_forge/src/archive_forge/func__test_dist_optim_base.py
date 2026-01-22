import threading
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
def _test_dist_optim_base(self, optim_cls, *args, **kwargs):
    module1 = MyModule()
    module2 = MyModule()
    params = [module1.get_w(), module2.get_w()]
    local_optim = optim_cls(params, *args, **kwargs)
    old_w1 = module1.w.clone().detach()
    old_w2 = module2.w.clone().detach()
    g_cpu = torch.Generator()
    g_cpu.manual_seed(0)
    t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
    t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
    output1 = module1.forward(t2)
    output2 = module2.forward(output1)
    loss = torch.add(output2, t1).sum()
    loss.backward()
    local_optim.step()
    owner1 = 'worker%d' % ((self.rank + 1) % self.world_size)
    owner2 = 'worker%d' % ((self.rank + 2) % self.world_size)
    remote_module1 = rpc.remote(owner1, MyModule)
    remote_module2 = rpc.remote(owner2, MyModule)
    remote_param1 = remote_method(MyModule.get_w, remote_module1)
    remote_param2 = remote_method(MyModule.get_w, remote_module2)
    old_w1_remote = remote_param1.to_here()
    self.assertEqual(old_w1, remote_param1.to_here())
    self.assertEqual(old_w2, remote_param2.to_here())
    dist_optim = DistributedOptimizer(optim_cls, [remote_param1, remote_param2], *args, **kwargs)
    with dist_autograd.context() as context_id:
        g_cpu.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        output1 = rpc_async_method(MyModule.forward, remote_module1, t2)
        output2 = rpc_async_method(MyModule.forward, remote_module2, output1.wait())
        loss = torch.add(output2.wait(), t1)
        dist_autograd.backward(context_id, [loss.sum()])
        dist_optim.step(context_id)
        new_w1 = rpc_async_method(MyModule.get_w, remote_module1).wait()
        new_w2 = rpc_async_method(MyModule.get_w, remote_module2).wait()
        self.assertNotEqual(old_w1, new_w1)
        self.assertNotEqual(old_w2, new_w2)
        self.assertEqual(new_w1, module1.get_w())
        self.assertEqual(new_w2, module2.get_w())