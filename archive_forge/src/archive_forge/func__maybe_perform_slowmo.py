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
def _maybe_perform_slowmo(self, optimizer: torch.optim.Optimizer) -> None:
    slowmo_rec = self._create_event_recorder('Slowmo')
    if self._should_perform_slowmo():
        self._global_momentum_step(optimizer)
    slowmo_rec.stop()
    self.logger.debug('Global momentum step completed')