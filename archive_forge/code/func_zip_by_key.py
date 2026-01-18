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
def zip_by_key(a: Dict[TK, TVa], b: Dict[TK, TVb]) -> Iterator[Tuple[TK, TVa, TVb]]:
    for arg, value in a.items():
        if arg in b:
            yield (arg, value, b[arg])