from collections import OrderedDict
import copy
import io
from itertools import chain
import logging
from math import inf
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.autograd import profiler
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import SGD, Optimizer
from fairscale.internal.params import calc_grad_norm, get_global_rank, recursive_copy_to_device
from fairscale.nn.misc import ParamBucket
@torch.no_grad()
def _broadcast_params(self) -> None:
    """Helper function to broadcast all the parameters from a given device"""
    with profiler.record_function('fairscale::oss::refresh_trainable'):
        if torch.device('cuda').type == self._default_device.type:
            for device in self._per_device_params.keys():
                torch.cuda.synchronize(device=device)
        work_handles = []
        if self.broadcast_fp16:
            for device in self.buckets.keys():
                for dst_rank, bucket in self.buckets[device].items():
                    bucket.to(dtype=torch.float16, device=device, non_blocking=True, keep_param_alignment=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        for device in self.buckets.keys():
            for dst_rank, bucket in self.buckets[device].items():
                work_handles.append(dist.broadcast(tensor=bucket.buffer, src=self._local_to_global_rank[dst_rank], group=self.group, async_op=True))
        _ = list(filter(lambda x: x.wait(), work_handles))
        if self.broadcast_fp16:
            for device in self.buckets.keys():
                for dst_rank, bucket in self.buckets[device].items():
                    bucket.to(dtype=torch.float32, device=device, non_blocking=True, keep_param_alignment=True)