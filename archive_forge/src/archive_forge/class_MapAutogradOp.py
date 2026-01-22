import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import AOTConfig, create_joint
from torch._functorch.eager_transforms import (
from torch._higher_order_ops.cond import (
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
class MapAutogradOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fw_graph, joint_graph, num_mapped_args, *flat_args):
        ctx.save_for_backward(*flat_args)
        ctx._joint_graph = joint_graph
        ctx._num_mapped_args = num_mapped_args
        with torch._C._AutoDispatchBelowAutograd():
            return (*map_impl(fw_graph, num_mapped_args, *flat_args),)

    @staticmethod
    def backward(ctx, *flat_grads):
        fw_args = ctx.saved_tensors
        fw_mapped_args = fw_args[:ctx._num_mapped_args]
        pos_args = fw_args[ctx._num_mapped_args:]
        grads = map_impl(ctx._joint_graph, ctx._num_mapped_args + len(flat_grads), *fw_mapped_args, *flat_grads, *pos_args)
        return (None, None, None, *grads)