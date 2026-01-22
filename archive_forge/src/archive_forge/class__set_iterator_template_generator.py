import itertools
import logging
import sys
import builtins
from contextlib import nullcontext
from pyomo.common.errors import TemplateExpressionError
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
class _set_iterator_template_generator(object):
    """Replacement iterator that returns IndexTemplates

    In order to generate template expressions, we hijack the normal Set
    iteration mechanisms so that this iterator is returned instead of
    the usual iterator.  This iterator will return IndexTemplate
    object(s) instead of the actual Set items the first time next() is
    called.
    """

    def __init__(self, _set, context):
        self._set = _set
        self.context = context

    def __iter__(self):
        return self

    def __next__(self):
        if self.context is None:
            raise StopIteration()
        context, self.context = (self.context, None)
        _set = self._set
        if _set.is_expression_type():
            d = _reduce_template_to_component(_set).dimen
        else:
            d = _set.dimen
        grp = context.next_group()
        if d is None or type(d) is not int:
            idx = (IndexTemplate(_set, None, context.next_id(), grp),)
        else:
            idx = tuple((IndexTemplate(_set, i, context.next_id(), grp) for i in range(d)))
        context.cache.append(idx)
        if len(idx) == 1:
            return idx[0]
        else:
            return idx
    next = __next__