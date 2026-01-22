import inspect
import logging
import os
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Optional, Type, Union
import torch
from torch import Tensor, nn
from torch.autograd.profiler import EventList, record_function
from torch.profiler import ProfilerAction, ProfilerActivity, tensorboard_trace_handler
from torch.utils.hooks import RemovableHandle
from typing_extensions import override
from lightning_fabric.accelerators.cuda import is_cuda_available
from pytorch_lightning.profilers.profiler import Profiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
def _delete_profilers(self) -> None:
    if self.profiler is not None:
        self.profiler.__exit__(None, None, None)
        self._cache_functions_events()
        self.profiler = None
    if self._schedule is not None:
        self._schedule.reset()
    if self._parent_profiler is not None:
        self._parent_profiler.__exit__(None, None, None)
        self._parent_profiler = None
    if self._register is not None:
        self._register.__exit__(None, None, None)
        self._register = None