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
def _create_profilers(self) -> None:
    if self.profiler is not None:
        return
    if self._emit_nvtx:
        if self._parent_profiler is None:
            self._parent_profiler = torch.cuda.profiler.profile()
        self.profiler = self._create_profiler(torch.autograd.profiler.emit_nvtx)
    else:
        self._parent_profiler = None
        self.profiler = self._create_profiler(torch.profiler.profile if _KINETO_AVAILABLE else torch.autograd.profiler.profile)