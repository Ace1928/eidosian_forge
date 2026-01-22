from __future__ import annotations
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import ModuleType
from typing import (
import numpy as np
def __array_function__(self, func: Callable[..., Any], types: Iterable[type], args: Iterable[Any], kwargs: Mapping[str, Any]) -> Any:
    ...