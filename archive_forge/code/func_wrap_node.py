import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def wrap_node(x):
    if isinstance(x, SymNode) and x.constant is not None:
        return x.constant
    if x.is_int():
        return SymInt(x)
    elif x.is_float():
        return SymFloat(x)
    elif x.is_bool():
        return SymBool(x)
    else:
        raise AssertionError(f'unrecognized return type {x}')