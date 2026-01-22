from enum import Enum
import functools
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.modules import Module
from .gossiper import Gossiper, PushPull, PushSum
from .graph_manager import GraphManager
from .graph_manager import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from .mixing_manager import MixingManager, UniformMixing
from .utils import (
from .utils.cuda_metering import EventRecorder, create_event_recorder
def _fp16_fp32_iterator(self, optimizer: torch.optim.Optimizer, fp32_params: Optional[torch.Tensor]) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Iterator for those fp16 parameters which have a fp32 copy"""
    if hasattr(optimizer, '_amp_stash') and hasattr(optimizer._amp_stash, 'fp16_groups'):
        for p_fp16_group, p_fp32_group in zip(optimizer._amp_stash.fp16_groups, optimizer._amp_stash.fp32_from_fp16_groups):
            for p_fp16, p_fp32 in zip(p_fp16_group, p_fp32_group):
                yield (p_fp16, p_fp32)
    elif fp32_params is not None:
        if isinstance(fp32_params, dict):
            fp32_params_list = list(fp32_params.values())
            assert len(fp32_params_list) == 1
            fp32_params = fp32_params_list[0]
        if isinstance(fp32_params, list):
            for p, fp32_param in zip(self.parameters(), fp32_params):
                yield (p.view(-1), fp32_param)
        else:
            offset = 0
            for p in self.parameters():
                yield (p.view(-1), fp32_params[offset:offset + p.numel()])
                offset += p.numel()