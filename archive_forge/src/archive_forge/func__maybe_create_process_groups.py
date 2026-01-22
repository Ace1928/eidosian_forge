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
def _maybe_create_process_groups(self, process_rank: int, process_world_size: int, nprocs_per_node: int, global_group: Optional[torch.distributed.ProcessGroup], master_group: Optional[torch.distributed.ProcessGroup], local_node_group: Optional[torch.distributed.ProcessGroup]) -> Tuple[int, int]:
    """Creates the process groups required for the SlowMo implementation"""
    self.local_rank = process_rank % self.nprocs_per_node
    assert process_world_size % self.nprocs_per_node == 0
    logical_world_size = process_world_size // self.nprocs_per_node
    logical_rank = process_rank // self.nprocs_per_node
    self._maybe_initialize_global_group(global_group, process_world_size)
    self._maybe_initialize_local_node_group(local_node_group, process_rank, logical_world_size)
    self._maybe_initialize_master_group(master_group, process_rank, process_world_size, nprocs_per_node)
    self.logger.debug('Initialization of all process groups complete')
    return (logical_rank, logical_world_size)