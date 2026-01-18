from __future__ import annotations
import contextlib
import functools
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Union
import torch
import torch.utils.hooks as hooks
from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches
from .hooks import AlignDevicesHook
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, PartialState
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
from .utils.constants import FSDP_PYTORCH_VERSION
from .utils.modeling import get_state_dict_offloaded_model
from .utils.other import is_compiled_module
from torch.distributed.algorithms.join import Join
@staticmethod
@contextmanager
def trigger_sync_in_backward(model):
    """Trigger the sync of the gradients in the next backward pass of the model after multiple forward passes under
        `Accelerator.no_sync` (only applicable in multi-GPU scenarios).

                If the script is not launched in distributed mode, this context manager does nothing.

                Args:
                    model (`torch.nn.Module`):
                        The model for which to trigger the gradient synchronization.

                Example:

                ```python
                >>> from accelerate import Accelerator

                >>> accelerator = Accelerator()
                >>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

                >>> with accelerator.no_sync():
                ...     loss_a = loss_func(model(input_a))  # first forward pass
                ...     loss_b = loss_func(model(input_b))  # second forward pass
                >>> accelerator.backward(loss_a)  # No synchronization across processes, only accumulate gradients
                >>> with accelerator.trigger_sync_in_backward(model):
                ...     accelerator.backward(loss_b)  # Synchronization across all processes
                >>> optimizer.step()
                >>> optimizer.zero_grad()
                ```
        """
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        yield
        return
    old_require_backward_grad_sync = model.require_backward_grad_sync
    old_require_forward_param_sync = model.require_forward_param_sync
    model.require_backward_grad_sync = True
    model.require_forward_param_sync = True
    model.reducer.prepare_for_backward([])
    try:
        yield
    finally:
        model.require_backward_grad_sync = old_require_backward_grad_sync
        model.require_forward_param_sync = old_require_forward_param_sync