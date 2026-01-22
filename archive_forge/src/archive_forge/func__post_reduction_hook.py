import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import os
import time
import traceback
import typing
from typing import (
import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairscale.internal.containers import apply_to_tensors
from fairscale.internal.parallel import (
from fairscale.internal.params import calc_grad_norm, recursive_copy_to_device
from fairscale.internal.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.internal.state_dict import replace_by_prefix_
from fairscale.nn.misc import FlattenParamsWrapper, _enable_pre_load_state_dict_hook
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from . import fsdp_optim_utils as ou
def _post_reduction_hook(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
    """Hook to call on each param after the reduce-scatter."""
    assert torch.cuda.current_stream() == self._streams['post_backward']
    self.assert_state(TrainingState.BACKWARD_POST)
    if self.gradient_postdivide_factor > 1:
        reduced_grad.data.div_(self.gradient_postdivide_factor)
    if self.mixed_precision:
        orig_param_grad_data = reduced_grad.data
        reduced_grad.data = reduced_grad.data.to(dtype=param.data.dtype)
        orig_param_grad_data.record_stream(torch.cuda.current_stream())
    if param._is_sharded:
        if getattr(param, '_saved_grad_shard', None) is None:
            param._saved_grad_shard = reduced_grad.data
        else:
            assert param._saved_grad_shard.shape == reduced_grad.shape, f'{param._saved_grad_shard.shape} vs {reduced_grad.shape}'
            param._saved_grad_shard.data += reduced_grad.data
        reduced_grad = param._saved_grad_shard.data
    if self.move_grads_to_cpu:
        param._cpu_grad.copy_(reduced_grad.data, non_blocking=True)
        reduced_grad.data.record_stream(torch.cuda.current_stream())