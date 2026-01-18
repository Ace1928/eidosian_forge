import builtins
import collections
import dataclasses
import inspect
import os
import sys
from array import array
from collections import Counter, UserDict, UserList, defaultdict, deque
from dataclasses import dataclass, fields, is_dataclass
from inspect import isclass
from itertools import islice
from types import MappingProxyType
from typing import (
from pip._vendor.rich.repr import RichReprResult
from . import get_console
from ._loop import loop_last
from ._pick import pick_bool
from .abc import RichRenderable
from .cells import cell_len
from .highlighter import ReprHighlighter
from .jupyter import JupyterMixin, JupyterRenderable
from .measure import Measurement
from .text import Text
def to_repr(obj: Any) -> str:
    """Get repr string for an object, but catch errors."""
    if max_string is not None and _safe_isinstance(obj, (bytes, str)) and (len(obj) > max_string):
        truncated = len(obj) - max_string
        obj_repr = f'{obj[:max_string]!r}+{truncated}'
    else:
        try:
            obj_repr = repr(obj)
        except Exception as error:
            obj_repr = f'<repr-error {str(error)!r}>'
    return obj_repr