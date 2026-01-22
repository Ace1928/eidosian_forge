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
class DeprecatedFieldDescriptor:
    """
    Descriptor for deprecated fields in an enum class.

    Args:
        field_name (`str`):
            The name of the deprecated field.
        replaced_with (`str`):
            The name of the field that replaces the deprecated one.
    """

    def __init__(self, field_name, replaced_with):
        self.field_name = field_name
        self.replaced_with = replaced_with

    def __get__(self, instance, owner):
        warnings.warn(f'The `{self.field_name}` of `{owner}` is deprecated and will be removed in v1.0.0. Please use the `{self.replaced_with}` instead.', FutureWarning)
        return getattr(owner, self.replaced_with)