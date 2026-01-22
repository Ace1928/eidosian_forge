import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def _target_to_str(self, target: Target) -> str:
    if callable(target):
        op = target.__name__
    else:
        assert isinstance(target, str)
        op = target
        if _is_magic(op):
            op = op[2:-2]
    op = _snake_case(op)
    return op