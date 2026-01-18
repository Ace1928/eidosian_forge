from __future__ import annotations
from typing import Type
from . import exc as orm_exc
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .. import exc
from .. import inspect
from ..sql import and_
from ..sql import operators
from ..sql.sqltypes import Integer
from ..sql.sqltypes import Numeric
from ..util import warn_deprecated
def visit_custom_op_binary_op(self, operator, eval_left, eval_right, clause):
    if operator.python_impl:
        return self._straight_evaluate(operator, eval_left, eval_right, clause)
    else:
        raise UnevaluatableError(f"Custom operator {operator.opstring!r} can't be evaluated in Python unless it specifies a callable using `.python_impl`.")