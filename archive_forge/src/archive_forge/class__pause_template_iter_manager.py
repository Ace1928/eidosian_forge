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
class _pause_template_iter_manager(object):
    __slots__ = ('iter_manager',)

    def __init__(self, iter_manager):
        self.iter_manager = iter_manager

    def __enter__(self):
        self.iter_manager.release()
        return self

    def __exit__(self, et, ev, tb):
        self.iter_manager.acquire()