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
def _perform_local_optimization(self, optimizer: torch.optim.Optimizer) -> None:
    """Performs the slow momentum on the local shard"""
    assert self.portion_start is not None
    with torch.no_grad():
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
                    current_p_gmb = self.global_momentum_buffer[buffer_start:buffer_end]
                    current_p_old = self.old_params[buffer_start:buffer_end]
                    current_p_gmb.mul_(self.slowmo_momentum).sub_(current_p, alpha=1 / group['lr']).add_(current_p_old, alpha=1 / group['lr'])
                    current_p_old.add_(current_p_gmb, alpha=-group['lr'] * self.slowmo_lr)
                    current_p.copy_(current_p_old)
                offset += numel