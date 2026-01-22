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
def _linear_presolve(self, comp_by_linear_var, lcon_by_linear_nnz, var_bounds):
    eliminated_vars = {}
    eliminated_cons = set()
    if not self.config.linear_presolve:
        return (eliminated_cons, eliminated_vars)
    for expr, info, _ in self.subexpression_cache.values():
        if not info.linear:
            continue
        expr_id = id(expr)
        for _id in info.linear:
            comp_by_linear_var[_id].append((expr_id, info))
    fixed_vars = [_id for _id, (lb, ub) in var_bounds.items() if lb == ub and lb is not None]
    var_map = self.var_map
    substitutions_by_linear_var = defaultdict(set)
    template = self.template
    one_var = lcon_by_linear_nnz[1]
    two_var = lcon_by_linear_nnz[2]
    while 1:
        if fixed_vars:
            _id = fixed_vars.pop()
            a = x = None
            b, _ = var_bounds[_id]
            logger.debug('NL presolve: bounds fixed %s := %s', var_map[_id], b)
            eliminated_vars[_id] = AMPLRepn(b, {}, None)
        elif one_var:
            con_id, info = one_var.popitem()
            expr_info, lb = info
            _id, coef = expr_info.linear.popitem()
            a = x = None
            b = expr_info.const = (lb - expr_info.const) / coef
            logger.debug('NL presolve: substituting %s := %s', var_map[_id], b)
            eliminated_vars[_id] = expr_info
            lb, ub = var_bounds[_id]
            if lb is not None and lb - b > TOL or (ub is not None and ub - b < -TOL):
                raise InfeasibleConstraintException(f"model contains a trivially infeasible variable '{var_map[_id].name}' (presolved to a value of {b} outside bounds [{lb}, {ub}]).")
            eliminated_cons.add(con_id)
        elif two_var:
            con_id, info = two_var.popitem()
            expr_info, lb = info
            _id, coef = expr_info.linear.popitem()
            id2, coef2 = expr_info.linear.popitem()
            id2_isdiscrete = var_map[id2].domain.isdiscrete()
            if var_map[_id].domain.isdiscrete() ^ id2_isdiscrete:
                if id2_isdiscrete:
                    _id, id2 = (id2, _id)
                    coef, coef2 = (coef2, coef)
            else:
                log_coef = _log10(abs(coef))
                log_coef2 = _log10(abs(coef2))
                if abs(log_coef2) < abs(log_coef) or (log_coef2 == -log_coef and log_coef2 < log_coef):
                    _id, id2 = (id2, _id)
                    coef, coef2 = (coef2, coef)
            a = -coef2 / coef
            x = id2
            b = expr_info.const = (lb - expr_info.const) / coef
            expr_info.linear[x] = a
            substitutions_by_linear_var[x].add(_id)
            eliminated_vars[_id] = expr_info
            logger.debug('NL presolve: substituting %s := %s*%s + %s', var_map[_id], a, var_map[x], b)
            x_lb, x_ub = var_bounds[x]
            lb, ub = var_bounds[_id]
            if lb is not None:
                lb = (lb - b) / a
            if ub is not None:
                ub = (ub - b) / a
            if a < 0:
                lb, ub = (ub, lb)
            if x_lb is None or (lb is not None and lb > x_lb):
                x_lb = lb
            if x_ub is None or (ub is not None and ub < x_ub):
                x_ub = ub
            var_bounds[x] = (x_lb, x_ub)
            if x_lb == x_ub and x_lb is not None:
                fixed_vars.append(x)
            eliminated_cons.add(con_id)
        else:
            return (eliminated_cons, eliminated_vars)
        for con_id, expr_info in comp_by_linear_var[_id]:
            c = expr_info.linear.pop(_id, 0)
            expr_info.const += c * b
            if x in expr_info.linear:
                expr_info.linear[x] += c * a
            elif a:
                expr_info.linear[x] = c * a
                comp_by_linear_var[x].append((con_id, expr_info))
                continue
            nnz = len(expr_info.linear)
            _old = lcon_by_linear_nnz[nnz + 1]
            if con_id in _old:
                lcon_by_linear_nnz[nnz][con_id] = _old.pop(con_id)
        for resubst in substitutions_by_linear_var.pop(_id, ()):
            expr_info = eliminated_vars[resubst]
            c = expr_info.linear.pop(_id, 0)
            expr_info.const += c * b
            if x in expr_info.linear:
                expr_info.linear[x] += c * a
            elif a:
                expr_info.linear[x] = c * a