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
def _queue_wait_for_post_backward(self) -> None:
    """Try to queue a `wait_for_post_backward` callback.

        Only called on root and only queue one callback at the beginning of
        outer most backward.
        """
    assert self._is_root
    if not self._post_backward_callback_queued:
        self.assert_state([TrainingState.IDLE])
        self._post_backward_callback_queued = True
        Variable._execution_engine.queue_callback(self._wait_for_post_backward)