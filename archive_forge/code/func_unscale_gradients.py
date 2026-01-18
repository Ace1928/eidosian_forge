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
def unscale_gradients(self, optimizer=None):
    """
        Unscale the gradients in mixed precision training with AMP. This is a noop in all other settings.

        Likely should be called through [`Accelerator.clip_grad_norm_`] or [`Accelerator.clip_grad_value_`]

        Args:
            optimizer (`torch.optim.Optimizer` or `list[torch.optim.Optimizer]`, *optional*):
                The optimizer(s) for which to unscale gradients. If not set, will unscale gradients on all optimizers
                that were passed to [`~Accelerator.prepare`].

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer = accelerator.prepare(model, optimizer)
        >>> outputs = model(inputs)
        >>> loss = loss_fn(outputs, labels)
        >>> accelerator.backward(loss)
        >>> accelerator.unscale_gradients(optimizer=optimizer)
        ```
        """
    if self.native_amp and self.mixed_precision == 'fp16':
        if optimizer is None:
            optimizer = self._optimizers
        elif not isinstance(optimizer, (tuple, list)):
            optimizer = [optimizer]
        for opt in optimizer:
            while isinstance(opt, AcceleratedOptimizer):
                opt = opt.optimizer
            self.scaler.unscale_(opt)