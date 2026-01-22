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
@torch.no_grad()
def _prep_grads_for_backward(self) -> None:
    """Make sure p.grad is correctly prepared for the backward with
        right shape, device, accumulated values, etc.
        """
    for p in self.params:
        if p.grad is not None:
            if p.grad.device != p.data.device:
                p.grad = None
            elif p.grad.size() == p._orig_size:
                pass
            elif p.grad.size() == p._fp32_shard.shape:
                p._saved_grad_shard = p.grad.data
                p.grad = None
            else:
                raise AssertionError(f'unexpected grad shape: {p.grad.size()}')