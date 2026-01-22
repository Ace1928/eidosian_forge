from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
import torch
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics import Metric
from typing_extensions import NotRequired, Required
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER, LRScheduler, ProcessGroup, ReduceLROnPlateau
class _SizedIterable(Protocol):

    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator:
        pass