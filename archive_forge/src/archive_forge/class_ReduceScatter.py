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
class ReduceScatter:

    def __init__(self, op):
        if op != dist.ReduceOp.SUM:
            raise NotImplementedError('ReduceScatter only supports SUM on threaded pg for now.')
        self.op = op

    @torch.no_grad()
    def work(self, data):
        start_reduction = [False for _ in range(len(data))]
        for each_rank_data in data:
            assert len(each_rank_data[1]) == 1
            to_scatter = each_rank_data[1][0]
            for i in range(len(to_scatter)):
                dest_tensor_on_rank_i = data[i][0]
                assert len(dest_tensor_on_rank_i) == 1
                dst_tensor_device = dest_tensor_on_rank_i[0].device
                if not start_reduction[i]:
                    dest_tensor_on_rank_i[0].copy_(to_scatter[i].to(dst_tensor_device))
                    start_reduction[i] = True
                else:
                    dest_tensor_on_rank_i[0].add_(to_scatter[i].to(dst_tensor_device))