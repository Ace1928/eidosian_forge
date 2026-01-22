from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TypeVar
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import OVERLAP
from ... import types as sqltypes
from ... import util
from ...sql import expression
from ...sql import operators
from ...sql._typing import _TypeEngineArgument
def _bind_param(self, operator, obj, _assume_scalar=False, type_=None):
    if _assume_scalar or operator is operators.getitem:
        return expression.BindParameter(None, obj, _compared_to_operator=operator, type_=type_, _compared_to_type=self.type, unique=True)
    else:
        return array([self._bind_param(operator, o, _assume_scalar=True, type_=type_) for o in obj])