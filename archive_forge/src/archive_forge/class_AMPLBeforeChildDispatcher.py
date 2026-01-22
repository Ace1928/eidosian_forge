import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
class AMPLBeforeChildDispatcher(BeforeChildDispatcher):
    __slots__ = ()

    def __init__(self):
        self[MonomialTermExpression] = self._before_monomial
        self[LinearExpression] = self._before_linear
        self[SumExpression] = self._before_general_expression

    @staticmethod
    def _record_var(visitor, var):
        vm = visitor.var_map
        try:
            _iter = var.parent_component().values(visitor.sorter)
        except AttributeError:
            _iter = (var,)
        for v in _iter:
            if v.fixed:
                continue
            vm[id(v)] = v

    @staticmethod
    def _before_string(visitor, child):
        visitor.encountered_string_arguments = True
        ans = AMPLRepn(child, None, None)
        ans.nl = (visitor.template.string % (len(child), child), ())
        return (False, (_GENERAL, ans))

    @staticmethod
    def _before_var(visitor, child):
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(_id, child)
                return (False, (_CONSTANT, visitor.fixed_vars[_id]))
            _before_child_handlers._record_var(visitor, child)
        return (False, (_MONOMIAL, _id, 1))

    @staticmethod
    def _before_monomial(visitor, child):
        arg1, arg2 = child._args_
        if arg1.__class__ not in native_types:
            try:
                arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
            except (ValueError, ArithmeticError):
                return (True, None)
        if not arg1:
            if arg2.fixed:
                _id = id(arg2)
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(id(arg2), arg2)
                arg2 = visitor.fixed_vars[_id]
                if arg2 != arg2:
                    deprecation_warning(f'Encountered {arg1}*{arg2} in expression tree.  Mapping the NaN result to 0 for compatibility with the nl_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.4.3')
            return (False, (_CONSTANT, arg1))
        _id = id(arg2)
        if _id not in visitor.var_map:
            if arg2.fixed:
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(_id, arg2)
                return (False, (_CONSTANT, arg1 * visitor.fixed_vars[_id]))
            _before_child_handlers._record_var(visitor, arg2)
        return (False, (_MONOMIAL, _id, arg1))

    @staticmethod
    def _before_linear(visitor, child):
        var_map = visitor.var_map
        const = 0
        linear = {}
        for arg in child.args:
            if arg.__class__ is MonomialTermExpression:
                arg1, arg2 = arg._args_
                if arg1.__class__ not in native_types:
                    try:
                        arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
                    except (ValueError, ArithmeticError):
                        return (True, None)
                if not arg1:
                    if arg2.fixed:
                        arg2 = visitor.check_constant(arg2.value, arg2)
                        if arg2 != arg2:
                            deprecation_warning(f'Encountered {arg1}*{str(arg2.value)} in expression tree.  Mapping the NaN result to 0 for compatibility with the nl_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.4.3')
                    continue
                _id = id(arg2)
                if _id not in var_map:
                    if arg2.fixed:
                        if _id not in visitor.fixed_vars:
                            visitor.cache_fixed_var(_id, arg2)
                        const += arg1 * visitor.fixed_vars[_id]
                        continue
                    _before_child_handlers._record_var(visitor, arg2)
                    linear[_id] = arg1
                elif _id in linear:
                    linear[_id] += arg1
                else:
                    linear[_id] = arg1
            elif arg.__class__ in native_types:
                const += arg
            else:
                try:
                    const += visitor.check_constant(visitor.evaluate(arg), arg)
                except (ValueError, ArithmeticError):
                    return (True, None)
        if linear:
            return (False, (_GENERAL, AMPLRepn(const, linear, None)))
        else:
            return (False, (_CONSTANT, const))

    @staticmethod
    def _before_named_expression(visitor, child):
        _id = id(child)
        if _id in visitor.subexpression_cache:
            obj, repn, info = visitor.subexpression_cache[_id]
            if info[2]:
                if repn.linear:
                    return (False, (_MONOMIAL, next(iter(repn.linear)), 1))
                else:
                    return (False, (_CONSTANT, repn.const))
            return (False, (_GENERAL, repn.duplicate()))
        else:
            return (True, None)