import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def _reduce_scatter_base(self, output_tensor, input_tensor, opts=ReduceScatterOptions()):
    return ret_work(output_tensor)