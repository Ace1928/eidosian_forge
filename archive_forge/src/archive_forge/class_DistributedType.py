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
class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **MULTI_NPU** -- Distributed on multiple NPUs.
        - **MULTI_XPU** -- Distributed on multiple XPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **XLA** -- Using TorchXLA.
        - **TPU** -- This field will be deprecated in v0.27.0. Use XLA instead.
    """
    NO = 'NO'
    MULTI_CPU = 'MULTI_CPU'
    MULTI_GPU = 'MULTI_GPU'
    MULTI_NPU = 'MULTI_NPU'
    MULTI_XPU = 'MULTI_XPU'
    DEEPSPEED = 'DEEPSPEED'
    FSDP = 'FSDP'
    XLA = 'XLA'
    MEGATRON_LM = 'MEGATRON_LM'
    TPU = DeprecatedFieldDescriptor('TPU', 'XLA')