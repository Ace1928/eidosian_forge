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
class TemplateSumExpression(NumericExpression):
    """
    Expression to represent an unexpanded sum over one or more sets.
    """
    __slots__ = ('_iters', '_local_args_')
    PRECEDENCE = 1

    def __init__(self, args, _iters):
        assert len(args) == 1
        self._args_ = args
        self._iters = _iters

    def nargs(self):
        ans = 1
        for iterGroup in self._iters:
            ans *= len(iterGroup[0]._set)
        return ans

    @property
    def args(self):
        return _TemplateSumExpression_argList(self)

    @property
    def _args_(self):
        return _TemplateSumExpression_argList(self)

    @_args_.setter
    def _args_(self, args):
        self._local_args_ = args

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._iters)

    def getname(self, *args, **kwds):
        return 'SUM'

    def is_potentially_variable(self):
        if any((arg.is_potentially_variable() for arg in self._local_args_ if arg.__class__ not in nonpyomo_leaf_types)):
            return True
        return False

    def _is_fixed(self, values):
        return all(values)

    def _compute_polynomial_degree(self, result):
        if None in result:
            return None
        return result[0]

    def _apply_operation(self, result):
        return sum(result)

    def _to_string(self, values, verbose, smap):
        ans = ''
        val = values[0]
        if val[0] == '(' and val[-1] == ')' and _balanced_parens(val[1:-1]):
            val = val[1:-1]
        iterStrGenerator = ((', '.join((str(i) for i in iterGroup)), iterGroup[0]._set.to_string(verbose=verbose) if hasattr(iterGroup[0]._set, 'to_string') else str(iterGroup[0]._set)) for iterGroup in self._iters)
        if verbose:
            iterStr = ', '.join(('iter(%s, %s)' % x for x in iterStrGenerator))
            return 'templatesum(%s, %s)' % (val, iterStr)
        else:
            iterStr = ' '.join(('for %s in %s' % x for x in iterStrGenerator))
            return 'SUM(%s %s)' % (val, iterStr)

    def _resolve_template(self, args):
        return SumExpression(args)