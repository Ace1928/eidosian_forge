import functools
import sys
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import torch
from torch import fx
class CompiledFn(Protocol):

    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ...