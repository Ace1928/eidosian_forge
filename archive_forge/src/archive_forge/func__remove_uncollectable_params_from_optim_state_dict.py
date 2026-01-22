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
def _remove_uncollectable_params_from_optim_state_dict(self, osd: Dict) -> Dict:
    """Return a new state dict filtering out the ones like MoE layers, which has
        ``no_broadcast_optim_state`` flag set.

        We also make rooms for the optimizer state on rank 0.

        Args:
            osd (Dict):
                Optimizer state dict from a rank. osd["state"] is what we mainly
                care. Osd may contain other keys and values, we need to keep. Therefore,
                we only change osd["state"] and not returning a new copy of osd
                which is slower and may also lose extra fields, like "loss_scale"
                used by fairseq.
        """
    for _, bufs in osd['state'].items():
        if 'step' in bufs.keys():
            assert type(bufs['step']) is int or ou.is_singleton_tensor(bufs['step'])
            if ou.is_singleton_tensor(bufs['step']):
                bufs['step'] = bufs['step'].item()
    uncollected_ids = [i for i, m in enumerate(get_fsdp_instances(self)) if m.no_broadcast_optim_state]
    new_state_value = {k: v for k, v in osd['state'].items() if k not in uncollected_ids}
    if self.rank == 0:
        self.uncollected_opt_state = {k: recursive_copy_to_device(v, non_blocking=False, device=torch.device('cpu')) for k, v in osd['state'].items() if k in uncollected_ids}
    osd['state'] = new_state_value
    return osd