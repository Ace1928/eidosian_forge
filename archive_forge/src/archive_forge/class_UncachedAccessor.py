from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
class UncachedAccessor(Generic[_Accessor]):
    """Acts like a property, but on both classes and class instances

    This class is necessary because some tools (e.g. pydoc and sphinx)
    inspect classes for which property returns itself and not the
    accessor.
    """

    def __init__(self, accessor: type[_Accessor]) -> None:
        self._accessor = accessor

    @overload
    def __get__(self, obj: None, cls) -> type[_Accessor]:
        ...

    @overload
    def __get__(self, obj: object, cls) -> _Accessor:
        ...

    def __get__(self, obj: None | object, cls) -> type[_Accessor] | _Accessor:
        if obj is None:
            return self._accessor
        return self._accessor(obj)