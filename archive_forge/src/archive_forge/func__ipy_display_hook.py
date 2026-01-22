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
def _ipy_display_hook(value: Any, console: Optional['Console']=None, overflow: 'OverflowMethod'='ignore', crop: bool=False, indent_guides: bool=False, max_length: Optional[int]=None, max_string: Optional[int]=None, max_depth: Optional[int]=None, expand_all: bool=False) -> Union[str, None]:
    from .console import ConsoleRenderable
    if _safe_isinstance(value, JupyterRenderable) or value is None:
        return None
    console = console or get_console()
    with console.capture() as capture:
        if _safe_isinstance(value, ConsoleRenderable):
            console.line()
        console.print(value if _safe_isinstance(value, RichRenderable) else Pretty(value, overflow=overflow, indent_guides=indent_guides, max_length=max_length, max_string=max_string, max_depth=max_depth, expand_all=expand_all, margin=12), crop=crop, new_line_start=True, end='')
    return capture.get().rstrip('\n')