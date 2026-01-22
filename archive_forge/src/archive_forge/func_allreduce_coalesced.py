import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
    return ret_work(tensor_list)