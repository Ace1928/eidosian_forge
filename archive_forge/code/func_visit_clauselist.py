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
def visit_clauselist(self, clause):
    evaluators = [self.process(clause) for clause in clause.clauses]
    dispatch = f'visit_{clause.operator.__name__.rstrip('_')}_clauselist_op'
    meth = getattr(self, dispatch, None)
    if meth:
        return meth(clause.operator, evaluators, clause)
    else:
        raise UnevaluatableError(f'Cannot evaluate clauselist with operator {clause.operator}')