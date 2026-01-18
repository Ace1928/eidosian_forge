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
def wrap_int(self, num):
    assert type(num) is int
    import sympy
    return SymNode(sympy.Integer(num), self.shape_env, int, num, constant=num, fx_node=num)