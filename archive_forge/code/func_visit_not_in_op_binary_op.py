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
def visit_not_in_op_binary_op(self, operator, eval_left, eval_right, clause):
    return self._straight_evaluate(lambda a, b: a not in b if a is not _NO_OBJECT else None, eval_left, eval_right, clause)