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
def _print_r0(self, msg: str, restart: bool=False) -> None:
    """Debugging utility to print memory usage stats nicely on rank 0"""
    if restart:
        self._tstart = time.time()
    if self.rank == 0:
        gb_denom = 1024 ** 3
        logging.info(f'{msg} cur={torch.cuda.memory_allocated() / gb_denom: .4f} GB, max={torch.cuda.max_memory_allocated() / gb_denom: .4f} GB, t={time.time() - self._tstart: .1f}')