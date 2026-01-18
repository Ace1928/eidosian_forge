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
def visit_and_clauselist_op(self, operator, evaluators, clause):

    def evaluate(obj):
        for sub_evaluate in evaluators:
            value = sub_evaluate(obj)
            if value is _EXPIRED_OBJECT:
                return _EXPIRED_OBJECT
            if not value:
                if value is None or value is _NO_OBJECT:
                    return None
                return False
        return True
    return evaluate