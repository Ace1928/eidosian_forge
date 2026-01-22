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
def _maybe_pre_communicate_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    ef_rec = self._create_event_recorder('Error feedback')
    if self._should_use_error_feedback(fp16_fp32_list):
        with torch.no_grad():
            for p_fp16, p_fp32 in fp16_fp32_list:
                if self._should_allreduce_params():
                    p_fp16.div_(self.logical_world_size)
                    p_fp16.mul_(self.logical_world_size)
                p_fp32 -= p_fp16.float()
            if self.ef1 is not None:
                for idx, (_, p_fp32) in enumerate(fp16_fp32_list):
                    p_fp32 += self.ef1[idx]
                    p_fp32.div_(2)
    ef_rec.stop()
    self.logger.debug('Error feedback completed')