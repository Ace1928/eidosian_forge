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
class _TemplateSumExpression_argList(object):
    """A virtual list to represent the expanded SumExpression args

    This class implements a "virtual args list" for
    TemplateSumExpressions without actually generating the expanded
    expression.  It can be accessed either in "one-pass" without
    generating a list of template argument values (more efficient), or
    as a random-access list (where it will have to create the full list
    of argument values (less efficient).

    The instance can be used as a context manager to both lock the
    IndexTemplate values within this context and to restore their original
    values upon exit.

    It is (intentionally) not iterable.

    """

    def __init__(self, TSE):
        self._tse = TSE
        self._i = 0
        self._init_vals = None
        self._iter = self._get_iter()
        self._lock = None

    def __len__(self):
        return self._tse.nargs()

    def __getitem__(self, i):
        if self._i == i:
            self._set_iter_vals(next(self._iter))
            self._i += 1
        elif self._i is not None:
            self._iter = list(self._get_iter() if self._i else self._iter)
            self._set_iter_vals(self._iter[i])
        else:
            self._set_iter_vals(self._iter[i])
        return self._tse._local_args_[0]

    def __enter__(self):
        self._lock = self
        self._lock_iters()

    def __exit__(self, exc_type, exc_value, tb):
        self._unlock_iters()
        self._lock = None

    def _get_iter(self):
        _sets = tuple((iterGroup[0]._set for iterGroup in self._tse._iters))
        return itertools.product(*_sets)

    def _lock_iters(self):
        self._init_vals = tuple((tuple((it.lock(self._lock) for it in iterGroup)) for iterGroup in self._tse._iters))

    def _unlock_iters(self):
        self._set_iter_vals(self._init_vals)
        for iterGroup in self._tse._iters:
            for it in iterGroup:
                it.unlock(self._lock)

    def _set_iter_vals(self, val):
        for i, iterGroup in enumerate(self._tse._iters):
            if len(iterGroup) == 1:
                iterGroup[0].set_value(val[i], self._lock)
            else:
                for j, v in enumerate(val[i]):
                    iterGroup[j].set_value(v, self._lock)