from __future__ import annotations
import ast
from functools import (
from keyword import iskeyword
import tokenize
from typing import (
import numpy as np
from pandas.errors import UndefinedVariableError
import pandas.core.common as com
from pandas.core.computation.ops import (
from pandas.core.computation.parsing import (
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
def _maybe_downcast_constants(self, left, right):
    f32 = np.dtype(np.float32)
    if left.is_scalar and hasattr(left, 'value') and (not right.is_scalar) and (right.return_type == f32):
        name = self.env.add_tmp(np.float32(left.value))
        left = self.term_type(name, self.env)
    if right.is_scalar and hasattr(right, 'value') and (not left.is_scalar) and (left.return_type == f32):
        name = self.env.add_tmp(np.float32(right.value))
        right = self.term_type(name, self.env)
    return (left, right)