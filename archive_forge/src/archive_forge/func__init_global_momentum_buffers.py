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
def _init_global_momentum_buffers(self, optimizer: torch.optim.Optimizer) -> None:
    """Initializes the slow momentum buffers"""
    self.global_momentum_buffers_initialized = True
    if not self.slowmo:
        return
    total_elements = 0
    params_dtype = None
    for group in optimizer.param_groups:
        for p in group['params']:
            total_elements += p.numel()
            if params_dtype is None:
                params_dtype, params_device = (p.dtype, p.device)
            assert p.dtype == params_dtype == torch.float32
            assert p.device == params_device
    self.world_portion_length = (total_elements + self.slowmo_num_shards - 1) // self.slowmo_num_shards
    if not self.is_current_node_a_slowmo_shard:
        return
    self.portion_start = self.process_rank * self.world_portion_length if self.slowmo_memory_efficient else 0
    self.portion_end = min((self.process_rank + 1) * self.world_portion_length, total_elements) if self.slowmo_memory_efficient else total_elements
    self.old_params = torch.empty(self.world_portion_length, dtype=params_dtype).to(params_device).detach()
    offset = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            numel = p.numel()
            if offset + numel > self.portion_start and offset < self.portion_end:
                overall_start = max(self.portion_start, offset)
                overall_end = min(self.portion_end, offset + numel)
                p_start = overall_start - offset
                p_end = overall_end - offset
                buffer_start = overall_start - self.portion_start
                buffer_end = overall_end - self.portion_start
                current_p = p.view(-1)[p_start:p_end]
                current_p_old = self.old_params[buffer_start:buffer_end]
                current_p_old.copy_(current_p)
            offset += numel
    self.global_momentum_buffer = torch.zeros_like(self.old_params).detach()