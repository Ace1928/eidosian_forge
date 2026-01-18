import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharded_tensor import (
@_sharded_op_impl(torch.nn.init.uniform_)
def uniform_(types, args=(), kwargs=None, pg=None):
    """
    Fills the Tensor in tensor.local_shards with values drawn from the uniform
    distribution :math:`\\mathcal{U}(a, b)`.
    Args:
        tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    validate_param(kwargs, 'kwargs')
    sharded_tensor = kwargs['tensor']
    validate_param(sharded_tensor, 'tensor')
    a = kwargs['a']
    validate_param(a, 'a')
    b = kwargs['b']
    validate_param(b, 'b')
    for shard in sharded_tensor.local_shards():
        torch.nn.init.uniform_(shard.tensor, a=a, b=b)
    return sharded_tensor