import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
def _extract_dist_info(self) -> None:
    """
        Extract the process group and device information from the joinables.

        If there are multiple joinables, then the context manager uses the
        first specified device.

        Preconditions:
            ``self._joinables`` is not ``None`` and is non-empty.

        Raises:
            ValueError
                If there are multiple conflicting ``process_group`` attributes
                among the ``Joinable`` objects.
        """
    process_group = None
    device = None
    for joinable in self._joinables:
        if process_group is None:
            process_group = joinable.join_process_group
        elif process_group != joinable.join_process_group:
            raise ValueError('Using join context manager with multiple process groups')
        if device is None:
            device = joinable.join_device
    self._process_group = process_group
    self._rank = dist.get_rank(self._process_group)
    self._device = device