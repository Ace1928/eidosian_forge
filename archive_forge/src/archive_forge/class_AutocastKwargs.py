import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class AutocastKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how `torch.autocast` behaves. Please refer to the
    documentation of this [context manager](https://pytorch.org/docs/stable/amp.html#torch.autocast) for more
    information on each argument.

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import AutocastKwargs

    kwargs = AutocastKwargs(cache_enabled=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """
    enabled: bool = True
    cache_enabled: bool = None