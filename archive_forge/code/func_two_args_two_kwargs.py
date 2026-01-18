from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@torch.jit.script
def two_args_two_kwargs(first_arg, second_arg, first_kwarg=torch.tensor([3, 3]), second_kwarg=torch.tensor([4, 4])):
    return first_arg + second_arg + first_kwarg + second_kwarg