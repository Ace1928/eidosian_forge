import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import (
from torch.jit._recursive import (
from torch.jit._state import (
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location
class _ScriptProfileColumn:

    def __init__(self, header: str, alignment: int=4, offset: int=0):
        self.header = header
        self.alignment = alignment
        self.offset = offset
        self.rows: Dict[int, Any] = {}

    def add_row(self, lineno: int, value: Any):
        self.rows[lineno] = value

    def materialize(self):
        max_length = len(self.header)
        rows: List[Tuple[int, str]] = []
        for key, value in self.rows.items():
            cell = str(value)
            rows.append((key, cell))
            max_length = max(len(cell), max_length)
        if self.alignment > 0:
            padding = max_length + self.alignment
            padding -= padding % self.alignment
        else:
            padding = 0
        rows = [(key, pad(cell, padding, self.offset)) for key, cell in rows]
        return (pad(self.header, padding, self.offset), rows)