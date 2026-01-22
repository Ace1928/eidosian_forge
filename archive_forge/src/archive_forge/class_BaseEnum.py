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
class BaseEnum(enum.Enum, metaclass=EnumWithContains):
    """An enum class that can get the value of an item with `str(Enum.key)`"""

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        """Method to list all the possible items in `cls`"""
        return list(map(str, cls))