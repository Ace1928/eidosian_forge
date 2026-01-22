import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
def _notify_procs_to_terminate(self):
    """Schedule an all-reduce to notify non-joined processes to terminate.

        Also raise a ``RuntimeError`` indicating that the current process has exhausted its inputs.
        """
    ones = torch.ones(1, device=self._device)
    dist.all_reduce(ones, group=self._process_group)
    raise RuntimeError(f'Rank {self._rank} exhausted all inputs.')