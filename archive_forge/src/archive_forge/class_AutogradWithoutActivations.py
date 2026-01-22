from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from .checkpoint import Checkpointing
from .messages import Transport
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors, Tensors
from .worker import Task
class AutogradWithoutActivations(torch.autograd.Function):
    """A helper class to add another edge in the autograd graph which allows us
    to delete the potentially large activations and still perform a backward
    pass. Returns return a phony tensor which is connected to the graph."""

    @staticmethod
    def forward(ctx, *x):
        return torch.tensor(1.0)

    @staticmethod
    def backward(ctx, grad):
        assert ctx.grad_from_pipeline is not None
        return ctx.grad_from_pipeline