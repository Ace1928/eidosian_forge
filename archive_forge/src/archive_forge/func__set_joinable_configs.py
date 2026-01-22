import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
def _set_joinable_configs(self) -> None:
    """Set the :class:`_JoinConfig` of each participating :class:`Joinable`."""
    assert len(self._joinables) > 0
    is_first_joinable = True
    for joinable in self._joinables:
        joinable._join_config = _JoinConfig(enable=self._enable, throw_on_early_termination=self._throw_on_early_termination, is_first_joinable=is_first_joinable)
        is_first_joinable = False