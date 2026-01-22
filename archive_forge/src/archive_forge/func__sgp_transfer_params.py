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
def _sgp_transfer_params(self, mix: bool=True) -> bool:
    """Transfers COPY of model parameters to gossip queue"""
    if not self.gossip_enable or self.process_rank % self.nprocs_per_node != 0:
        return False
    self.logger.debug('transferring model params')
    if not self.params_mixed:
        self.logger.warning('params not mixed')
        return False
    mix = mix and (not self.lazy_mixing)
    self._sgp_ps_numerator()
    if mix:
        self.ps_weight *= self.gossip_ps_factor
    self.gossip_ps_weight.copy_(self.ps_weight)
    with torch.no_grad():
        for p, gossip_device_buffer_elem in zip(self.module.parameters(), self.gossip_device_buffer):
            if mix:
                p.mul_(cast(torch.Tensor, self.gossip_ps_factor.type(p.dtype)))
            gossip_device_buffer_elem.copy_(p)
    self.gossip_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(self.gossip_stream):
        for b, gp in zip(self.gossip_device_buffer, self.gossip_params):
            gp.copy_(b, non_blocking=True)
    self.logger.debug('transferred model params')
    self.params_mixed = False
    self.gossiping = True
    self.train_flag.set()
    return True