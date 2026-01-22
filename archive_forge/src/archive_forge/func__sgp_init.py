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
def _sgp_init(self, module: torch.nn.Module, first_param_dtype: torch.dtype, logical_rank: int, logical_world_size: int, comm_device: Optional[torch.device]=None, graph: Optional[GraphManager]=None, mixing: Optional[MixingManager]=None, push_sum: bool=True, overlap: bool=False, synch_freq: int=0, use_streams: bool=True, slowmo_sgp_average_params: bool=False) -> None:
    """Perform initialization for Stochastic Gradient Push base algorithm"""
    if graph is None:
        graph = NPDDEGraph(logical_rank, logical_world_size, self.nprocs_per_node, self.local_rank)
    if mixing is None:
        mixing = UniformMixing(graph, comm_device)
    self.dist_config.update({'graph': graph, 'mixing': mixing, 'push_sum': push_sum})
    self.overlap = overlap
    assert not self.overlap
    self.synch_freq = synch_freq
    self.asynch = synch_freq > 0
    self.ps_weight = torch.ones(1, device=comm_device, dtype=first_param_dtype)
    self.is_sgp_ps_numerator = False
    self.gossip_enable = True
    self.gossiping = False
    self.params_mixed = True
    self.gossip_ps_factor = torch.zeros(1, device=comm_device, dtype=first_param_dtype)
    self.gossip_ps_weight = self.ps_weight.clone()
    self.gossip_params = []
    self.gossip_device_buffer = []
    for p in module.parameters():
        cp = cast(torch.nn.Parameter, p.clone().detach_())
        cp = cast(torch.nn.Parameter, cp.cpu().pin_memory() if self._cpu_comm else cp.cuda())
        self.gossip_params.append(cp)
        self.gossip_device_buffer.append(cp)
    self.gossip_lock = threading.Lock()
    self.gossip_flag = threading.Event()
    self.train_flag = threading.Event()
    if cast(torch.device, self.dist_config['comm_device']).type != 'cpu' and use_streams:
        self.gossip_stream = torch.cuda.Stream()
    else:
        self.gossip_stream = torch.cuda.current_stream()
    if self.process_rank % self.nprocs_per_node == 0:
        self.gossip_thread = threading.Thread(target=SlowMoDistributedDataParallel._sgp_gossip_target, args=(self.dist_config, self.gossip_flag, self.train_flag, self.gossip_lock, self.gossip_params, self.gossip_device_buffer, self.gossip_ps_weight, self.gossip_ps_factor, self.gossip_stream))
        self.gossip_thread.daemon = True
        self.gossip_thread.name = 'Gossip-Thread'
        self.gossip_thread.start()
    else:
        self.gossip_flag.set()
    self.gossip_flag.wait()
    self.gossip_flag.clear()
    self.lazy_mixing = not self.asynch and cast(MixingManager, self.dist_config['mixing']).is_regular()
    self.lazy_ps_factor = self.gossip_ps_factor.clone()
    self.logger.debug('lazy mixing: %r', self.lazy_mixing)