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
def _sgp_query_gossip_queue(self, non_blocking: bool=False) -> bool:
    """Check gossip-queue for push-sum residuals and update model"""
    if not self.gossip_enable:
        return False
    self.logger.debug('querying gossip queue')
    if not self.gossiping:
        if self.process_rank % self.nprocs_per_node == 0:
            self.logger.warning('not gossiping right now')
        return False
    if not non_blocking and (not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT)):
        raise RuntimeError('Gossip flag timeout')
        sys.exit()
    if self.gossip_flag.is_set():
        self.logger.debug('received gossip flag')
        if self.gossip_ps_weight[0] == -1:
            self.gossip_flag.clear()
            self.params_mixed = True
            self.gossiping = False
            self._sgp_transfer_params(mix=False)
            return False
        self.lazy_ps_factor.copy_(self.gossip_ps_factor)
        self._sgp_ps_numerator()
        self.ps_weight += self.gossip_ps_weight
        if self.lazy_mixing:
            self.ps_weight *= self.lazy_ps_factor
        with torch.no_grad():
            for p, r in zip(self.module.parameters(), self.gossip_device_buffer):
                p.add_(r)
                if self.lazy_mixing:
                    p.mul_(cast(torch.Tensor, self.lazy_ps_factor.type(p.dtype)))
        self.logger.debug('updated ps-weight %f', self.ps_weight)
        self.logger.debug('updated model params')
        self.gossip_flag.clear()
        self.params_mixed = True
        self.gossiping = False
        return True
    return False