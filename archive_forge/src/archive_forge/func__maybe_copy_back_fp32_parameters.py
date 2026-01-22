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
def _maybe_copy_back_fp32_parameters(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    ef_copy_rec = self._create_event_recorder('Error feedback copy back')
    if (self._should_perform_sgp() or self._should_allreduce_params() or self._should_perform_slowmo()) and fp16_fp32_list:
        with torch.no_grad():
            for idx, (p_fp16, p_fp32) in enumerate(fp16_fp32_list):
                p_fp16.copy_(p_fp32)
    ef_copy_rec.stop()
    self.logger.debug('Error feedback copy-back completed')