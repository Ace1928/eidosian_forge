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
def _use_full_params(self) -> None:
    """
        Switch p.data pointers to use the full params.

        Note: this assumes full params are already gathered.

        Note: this might be called after full_params is already in used. So please
              make sure it is idempotent in that case.
        """
    assert self.has_full_params
    for p in self.params:
        if not p._is_sharded:
            if self.mixed_precision or self.move_params_to_cpu:
                assert p._fp16_shard is not None
                assert p._fp16_shard.storage().size() != 0
                p.data = p._fp16_shard
        else:
            assert p._full_param_padded.storage().size() != 0, f'{p._orig_size} {id(self)}'
            p.data = p._full_param_padded[:p._orig_size.numel()].view(p._orig_size)