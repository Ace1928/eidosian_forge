import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def _sp24_to_dense(self) -> torch.Tensor:
    e = torch.eye(self.shape[1], self.shape[1], device=self.device, dtype=self.dtype)
    return self @ e