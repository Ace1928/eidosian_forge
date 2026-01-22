import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
class AllReduce:

    def __init__(self, op):
        if op.op not in _reduce_ops:
            raise NotImplementedError(f'AllReduce op {op.op} not supported on multithreaded pg for now.')
        self.op = op.op

    @torch.no_grad()
    def work(self, data):
        for i in range(len(data[0])):
            tensors = []
            rank_0_device = data[0][i].device
            for src_rank in range(0, len(data)):
                tensors.append(data[src_rank][i].to(rank_0_device))
            res = _reduce_ops[self.op](tensors)
            for src_rank in range(len(data)):
                data[src_rank][i].copy_(res.to(data[src_rank][i].device))