from collections import deque
import contextlib
import functools
from itertools import chain
import logging
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Union
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd.profiler as profiler
import torch.distributed as dist
from fairscale.internal.params import Workhandle, get_global_rank
from fairscale.nn.misc import GradBucket
from fairscale.optim import OSS
def _setup_bucket_strategy(self) -> None:
    """Devise a bucketing strategy on a per-rank ownership level.
        These buckets will not be sharded, since the gradients would be re-allocated during the backward in that case.
        This method can be a slow for big models, but it it not typically called often (not for every forward for instance)
        """
    with profiler.record_function('fairscale::sdp::setup_buckets'):
        if not self._use_buckets:
            return
        self._buckets = {}
        self._should_bucket_grad = [False for _ in self._trainable_params]
        for i, param in enumerate(self._trainable_params):
            device = param.device
            dst_rank = self._trainable_param_to_rank[param]
            if param.device not in self._buckets.keys():
                self._buckets[param.device] = {}
            if dst_rank not in self._buckets[param.device].keys():
                self._buckets[param.device][dst_rank] = GradBucket(self._buffer_max_size, dtype=param.dtype, device=param.device, destination=self._local_to_global_rank[dst_rank])
            if self._buckets[device][dst_rank].can_add_grad_view(param):
                self._buckets[device][dst_rank].add_grad(param)
                self._should_bucket_grad[i] = True
        self._bucket_list = list(chain(*[self._buckets[device].values() for device in self._buckets.keys()]))
        for bucket in self._bucket_list:
            bucket.shrink()