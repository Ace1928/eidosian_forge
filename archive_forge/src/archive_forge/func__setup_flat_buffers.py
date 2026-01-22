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
def _setup_flat_buffers(self) -> None:
    """Make all params which are on the same device and tied to the same rank views of a single buffer.
        This is used at construction time, and anytime parameter trainability is changed (frozen or unfrozen) and
        `refresh_trainability` is called.
        """
    for device, per_rank_params in self._per_device_params.items():
        if device not in self.buckets.keys():
            self.buckets[device] = {}
        for dst_rank, params in enumerate(per_rank_params):
            if len(params) > 0:
                for param in filter(lambda x: not x.requires_grad, params):
                    param.data = param.data.detach().clone()
                trainable_params = list(filter(lambda x: x.requires_grad, params))
                if trainable_params:
                    buffer_size = sum(map(lambda x: x.numel(), trainable_params))
                    bucket = ParamBucket(size=buffer_size, dtype=trainable_params[0].dtype, device=device)
                    for param in trainable_params:
                        bucket.add_param(param)
                    self.buckets[device][dst_rank] = bucket
    devices_in_use = list(self._per_device_params.keys())
    devices_to_pop = list(filter(lambda x: x not in devices_in_use, self.buckets.keys()))
    for d in devices_to_pop:
        self.buckets.pop(d)