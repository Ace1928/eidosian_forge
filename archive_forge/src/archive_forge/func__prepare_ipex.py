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
def _prepare_ipex(self, *args):
    if not is_ipex_available():
        raise ImportError("IPEX is not installed or IPEX's version does not match current PyTorch version. Please refer to https://github.com/intel/intel-extension-for-pytorch.")
    else:
        import intel_extension_for_pytorch as ipex
    model = None
    optimizer = None
    result = [obj for obj in args]
    for obj in result:
        if isinstance(obj, torch.nn.Module):
            model = obj
            model.train()
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = obj
    if optimizer is not None and model is not None:
        dtype = torch.bfloat16 if self.state.mixed_precision == 'bf16' else None
        if self.device.type == 'xpu' and is_xpu_available():
            model = model.to(self.device)
            model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=dtype, inplace=True, level='O1')
        else:
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype, inplace=True, level='O1')
    for i in range(len(result)):
        if isinstance(result[i], torch.nn.Module):
            result[i] = model
        elif isinstance(result[i], torch.optim.Optimizer):
            result[i] = optimizer
    return tuple(result)