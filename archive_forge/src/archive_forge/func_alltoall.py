import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def alltoall(self, output_tensors: List[Tensor], input_tensors: List[Tensor], opts=AllToAllOptions()):
    return ret_work(output_tensors)