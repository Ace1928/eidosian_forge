import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import (
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
def _ensure_all_tensors_same_dtype(*tensors) -> None:
    last_dtype = None
    for tensor in itertools.chain(*map(_as_iterable, tensors)):
        tensor_dtype = tensor.dtype
        if tensor_dtype.is_complex:
            tensor_dtype = torch.float32 if tensor_dtype == torch.complex64 else torch.complex128
        if last_dtype is None:
            last_dtype = tensor_dtype
        elif last_dtype != tensor_dtype:
            raise ValueError(f'Invalid usage of tensors with different dtypesFound {last_dtype} and  {tensor.dtype}')