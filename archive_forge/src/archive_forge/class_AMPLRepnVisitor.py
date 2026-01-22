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
class AMPLRepnVisitor(StreamBasedExpressionVisitor):

    def __init__(self, template, subexpression_cache, subexpression_order, external_functions, var_map, used_named_expressions, symbolic_solver_labels, use_named_exprs, sorter):
        super().__init__()
        self.template = template
        self.subexpression_cache = subexpression_cache
        self.subexpression_order = subexpression_order
        self.external_functions = external_functions
        self.active_expression_source = None
        self.var_map = var_map
        self.used_named_expressions = used_named_expressions
        self.symbolic_solver_labels = symbolic_solver_labels
        self.use_named_exprs = use_named_exprs
        self.encountered_string_arguments = False
        self.fixed_vars = {}
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack
        self.sorter = sorter

    def check_constant(self, ans, obj):
        if ans.__class__ not in native_numeric_types:
            if ans is None:
                return InvalidNumber(None, f"'{obj}' evaluated to a nonnumeric value '{ans}'")
            if ans.__class__ is InvalidNumber:
                return ans
            elif ans.__class__ in native_complex_types:
                return complex_number_error(ans, self, obj)
            else:
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(ans, f"'{obj}' evaluated to a nonnumeric value '{ans}'")
        if ans != ans:
            return InvalidNumber(nan, f"'{obj}' evaluated to a nonnumeric value '{ans}'")
        return ans

    def cache_fixed_var(self, _id, child):
        val = self.check_constant(child.value, child)
        lb, ub = child.bounds
        if lb is not None and lb - val > TOL or (ub is not None and ub - val < -TOL):
            raise InfeasibleConstraintException(f"model contains a trivially infeasible variable '{child.name}' (fixed value {val} outside bounds [{lb}, {ub}]).")
        self.fixed_vars[_id] = self.check_constant(child.value, child)

    def initializeWalker(self, expr):
        expr, src, src_idx, self.expression_scaling_factor = expr
        self.active_expression_source = (src_idx, id(src))
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return (False, self.finalizeResult(result))
        return (True, expr)

    def beforeChild(self, node, child, child_idx):
        return _before_child_handlers[child.__class__](self, child)

    def enterNode(self, node):
        if node.__class__ in sum_like_expression_types:
            data = AMPLRepn(0, {}, None)
            data.nonlinear = []
            return (node.args, data)
        else:
            return (node.args, [])

    def exitNode(self, node, data):
        if data.__class__ is AMPLRepn:
            if data.linear or data.nonlinear or data.nl:
                return (_GENERAL, data)
            else:
                return (_CONSTANT, data.const)
        return _operator_handles[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        ans = node_result_to_amplrepn(result)
        ans.mult *= self.expression_scaling_factor
        if ans.nl is not None:
            if not ans.nl[1]:
                raise ValueError('Numeric expression resolved to a string constant')
            if not ans.linear:
                ans.named_exprs.update(ans.nl[1])
                ans.nonlinear = ans.nl
                ans.const = 0
            else:
                pass
            ans.nl = None
        if ans.nonlinear.__class__ is list:
            ans.compile_nonlinear_fragment(self)
        if not ans.linear:
            ans.linear = {}
        if ans.mult != 1:
            linear = ans.linear
            mult, ans.mult = (ans.mult, 1)
            ans.const *= mult
            if linear:
                for k in linear:
                    linear[k] *= mult
            if ans.nonlinear:
                if mult == -1:
                    prefix = self.template.negation
                else:
                    prefix = self.template.multiplier % mult
                ans.nonlinear = (prefix + ans.nonlinear[0], ans.nonlinear[1])
        self.active_expression_source = None
        return ans