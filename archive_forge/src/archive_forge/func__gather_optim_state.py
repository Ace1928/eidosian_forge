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
def _gather_optim_state(self, sd_state: Dict[int, Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, List]], Dict[int, Dict[str, List]]]:
    """For each value in state[i], if the value is a tensor, collect it from the world. Else use rank 0's entry."""
    gathered_state: Dict[int, Dict[str, List[Any]]] = {}
    singleton_state: Dict[int, Dict[str, List[Any]]] = {}
    fsdp_instances = get_fsdp_instances(self, skip_empty=True)
    assert len(fsdp_instances) >= len(sd_state), f'{len(fsdp_instances)} vs. {len(sd_state)}'
    for k, v in sd_state.items():
        gathered_state[k] = {}
        singleton_state[k] = {}
        non_shared_params = fsdp_instances[k].non_shared_params()
        non_shared_world_size = fsdp_instances[k].world_size
        non_shared_process_group = fsdp_instances[k].process_group
        assert len(non_shared_params) == 1, f'Only flatten param or a single non-shared param is supported: len={len(non_shared_params)} FSDP={self}'
        desired_buffer_size = non_shared_params[0]._full_param_padded.size()
        buffer = None
        singleton_buffer = None
        for buffer_name, t in v.items():
            if torch.is_tensor(t):
                t = t.to(self.compute_device)
            if ou.is_singleton_tensor(t):
                if singleton_buffer is None:
                    singleton_buffer = list(t.new_zeros(non_shared_world_size).chunk(non_shared_world_size))
                dist.all_gather(singleton_buffer, t, group=non_shared_process_group)
                if self.rank == 0:
                    singleton_state[k][buffer_name] = [x.cpu().squeeze() for x in singleton_buffer]
                    assert ou.is_singleton_tensor(singleton_state[k][buffer_name][0])
            elif torch.is_tensor(t):
                if buffer is None:
                    buffer = list(t.new_zeros(*desired_buffer_size).chunk(non_shared_world_size))
                dist.all_gather(buffer, t, group=non_shared_process_group)
                if self.rank == 0:
                    gathered_state[k][buffer_name] = [x.cpu() for x in buffer]
            elif self.rank == 0:
                gathered_state[k][buffer_name] = [t]
    return (gathered_state, singleton_state)