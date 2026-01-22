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
class RegisterRecordFunction:
    """While profiling autograd operations, this class will add labels for module names around the forward function.

    The Lightning PyTorch Profiler will activate this feature automatically. It can be deactivated as follows:

    Example::
        from pytorch_lightning.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(record_module_names=False)
        Trainer(profiler=profiler)

    It can be used outside of Lightning as follows:

    Example::
        from pytorch_lightning import Trainer, seed_everything
        with RegisterRecordFunction(model):
            out = model(batch)

    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._records: Dict[str, record_function] = {}
        self._handles: Dict[str, List[RemovableHandle]] = {}

    def _start_recording_forward(self, _: nn.Module, input: Tensor, record_name: str) -> Tensor:
        record = record_function('[pl][module]' + record_name)
        record.__enter__()
        self._records[record_name] = record
        return input

    def _stop_recording_forward(self, _: nn.Module, __: Tensor, output: Tensor, record_name: str) -> Tensor:
        self._records[record_name].__exit__(None, None, None)
        return output

    def __enter__(self) -> None:
        for module_name, module in self._model.named_modules():
            if module_name:
                full_name = f'{type(module).__module__}.{type(module).__name__}'
                record_name = f'{full_name}: {module_name}'
                pre_forward_handle = module.register_forward_pre_hook(partial(self._start_recording_forward, record_name=record_name))
                post_forward_handle = module.register_forward_hook(partial(self._stop_recording_forward, record_name=record_name))
                self._handles[module_name] = [pre_forward_handle, post_forward_handle]

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        for handles in self._handles.values():
            for h in handles:
                h.remove()
        self._handles = {}