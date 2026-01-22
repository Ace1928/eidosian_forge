import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def _handle_argument(self, value: Any, is_write: bool, name: Optional[str]=None, is_output: bool=False) -> None:
    if isinstance(value, torch.Tensor) and value.is_cuda:
        data_ptr = value.data_ptr()
        if is_write:
            self.dataptrs_written.add(data_ptr)
        else:
            self.dataptrs_read.add(data_ptr)
        self.tensor_aliases.setdefault(data_ptr, [])
        if name is not None:
            self.tensor_aliases[data_ptr].append(name)
        if is_output:
            self.outputs.add(data_ptr)