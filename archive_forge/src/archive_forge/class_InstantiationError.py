from __future__ import annotations
import dataclasses
import itertools
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, TypeVar, Union
from typing_extensions import get_args
from . import _arguments, _fields, _parsers, _resolver, _strings
from .conf import _markers
@dataclasses.dataclass(frozen=True)
class InstantiationError(Exception):
    """Exception raised when instantiation fail; this typically means that values from
    the CLI are invalid."""
    message: str
    arg: Union[_arguments.ArgumentDefinition, str]