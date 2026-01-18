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
def visit_unary(self, clause):
    eval_inner = self.process(clause.element)
    if clause.operator is operators.inv:

        def evaluate(obj):
            value = eval_inner(obj)
            if value is _EXPIRED_OBJECT:
                return _EXPIRED_OBJECT
            elif value is None:
                return None
            return not value
        return evaluate
    raise UnevaluatableError(f'Cannot evaluate {type(clause).__name__} with operator {clause.operator}')