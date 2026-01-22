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
class _wildcard_info(object):
    __slots__ = ('iter', 'source', 'value', 'original_value', 'objects')

    def __init__(self, src, obj):
        self.source = src
        self.original_value = obj._value
        self.objects = [obj]
        self.reset()
        if self.original_value in (None, _NotSpecified):
            self.advance()

    def advance(self):
        with _TemplateIterManager.pause():
            self.value = next(self.iter)
        for obj in self.objects:
            obj.set_value(self.value)

    def reset(self):
        with _TemplateIterManager.pause():
            self.iter = iter(self.source)

    def restore(self):
        for obj in self.objects:
            obj.set_value(self.original_value)