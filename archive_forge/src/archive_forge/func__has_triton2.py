import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def _has_triton2():
    if not _is_triton_available():
        return False
    import triton
    tv = TorchVersion(triton.__version__)
    return tv >= (2, 1) or tv == (2, 0)