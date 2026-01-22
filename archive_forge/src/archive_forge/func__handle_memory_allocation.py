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
def _handle_memory_allocation(self, data_ptr: DataPtr) -> None:
    self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
    stack_trace = traceback.StackSummary.extract(traceback.walk_stack(inspect.currentframe()), lookup_lines=False)
    stack_trace.reverse()
    self.tensors_accessed.create_tensor(data_ptr, stack_trace)