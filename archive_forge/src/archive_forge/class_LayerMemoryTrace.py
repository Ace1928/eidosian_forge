from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
class LayerMemoryTrace(NamedTuple):
    """
    Trace event providing the current memory usage at a point
    occuring during the forward or backward

        module_name: name of the module under processing
        module_params: size of the module parameters
        allocated: state of the PyTorch allocated memory
        reserved: state of the PyTorch reserved memory
        is_forward: whether the trace was collected during forward
        all_gathered: memory gathered since last event by FSDP
        cumul_all_gathered: total amount of memory currently gathered by FSDP
        event: additional information on the trace
    """
    module_name: str
    module_params: int
    allocated: int
    reserved: int
    is_forward: bool
    all_gathered: int
    cumul_all_gathered: int
    event: Union[TraceForwardEvent, TraceBackwardEvent]

    def to_dict(self) -> Dict[str, Any]:
        return {'module_name': self.module_name, 'module_params': self.module_params, 'allocated': self.allocated, 'reserved': self.reserved, 'is_forward': self.is_forward, 'all_gathered': self.all_gathered, 'cumul_all_gathered': self.cumul_all_gathered, 'event': self.event.to_dict()}

    @classmethod
    def from_dict(cls, serialized: Dict[str, Any]) -> 'LayerMemoryTrace':
        if serialized['is_forward']:
            event: Union[TraceForwardEvent, TraceBackwardEvent] = TraceForwardEvent.from_dict(serialized['event'])
        else:
            event = TraceBackwardEvent.from_dict(serialized['event'])
        return LayerMemoryTrace(module_name=serialized['module_name'], module_params=serialized['module_params'], allocated=serialized['allocated'], reserved=serialized['reserved'], is_forward=serialized['is_forward'], all_gathered=serialized['all_gathered'], cumul_all_gathered=serialized['cumul_all_gathered'], event=event)