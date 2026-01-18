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
def with_shape_env(self, shape_env: 'ShapeEnv') -> 'SymNode':
    return SymNode(self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node)