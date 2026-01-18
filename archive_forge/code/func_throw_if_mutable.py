import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
def throw_if_mutable(schema):
    if schema.is_mutable:
        raise RuntimeError(f'Tried to trace mutable operation {schema}. FX only supports functional code, so operations that mutate operands in-place (e.g. via `out` arguments) are not supported')