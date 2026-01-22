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
def append_shared_param(self, p: Parameter) -> None:
    """Add a param that's already owned by another FSDP wrapper.

            .. warning:: This is experimental!

            This only works with all sharing FSDP modules are un-flattened.

            p must to be already sharded by the owning module.

            Check the corresponding unit tests to see how is it used and tested.
            In particular, the sharing FSDP wrappers are "siblings" not "parent"
            and "child" of each other in the nested module structure.

        Args:
            p (Parameter):
                The shared parameter.
        """
    assert self._is_root is None
    assert not self.flatten_parameters
    assert isinstance(p, Parameter)
    assert p._is_sharded
    p._is_shared = True
    assert len(list(filter(lambda p: not (hasattr(p, '_is_shared') and p._is_shared), self.params))) > 0, 'Must have at least 1 non-shared param.'
    self.params.append(p)
    self._has_shared_params = True