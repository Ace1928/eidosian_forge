from typing import Callable, Dict, Tuple
from torch import nn
from torch.distributed import rpc
def DistributedLoss(loss: nn.Module, *args: Tuple, **kwargs: Dict) -> Callable:
    loss_func = loss(*args, **kwargs)

    def dloss(input_rref: rpc.RRef, target_rref: rpc.RRef) -> rpc.RRef:
        return rpc.remote(input_rref.owner(), _rloss, args=(loss_func, input_rref, target_rref))
    return dloss