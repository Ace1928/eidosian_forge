from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
class _BigM_MixIn(object):

    def _get_bigM_arg_list(self, bigm_args, block):
        arg_list = []
        if bigm_args is None:
            return arg_list
        while block is not None:
            if block in bigm_args:
                arg_list.append({block: bigm_args[block]})
            block = block.parent_block()
        return arg_list

    def _set_up_expr_bound_visitor(self):
        self._expr_bound_visitor = ExpressionBoundsVisitor(use_fixed_var_values_as_bounds=False)

    def _process_M_value(self, m, lower, upper, need_lower, need_upper, src, key, constraint, from_args=False):
        m = _convert_M_to_tuple(m, constraint)
        if need_lower and m[0] is not None:
            if from_args:
                self.used_args[key] = m
            lower = (m[0], src, key)
            need_lower = False
        if need_upper and m[1] is not None:
            if from_args:
                self.used_args[key] = m
            upper = (m[1], src, key)
            need_upper = False
        return (lower, upper, need_lower, need_upper)

    def _get_M_from_args(self, constraint, bigMargs, arg_list, lower, upper):
        if bigMargs is None:
            return (lower, upper)
        need_lower = constraint.lower is not None
        need_upper = constraint.upper is not None
        parent = constraint.parent_component()
        if constraint in bigMargs:
            m = bigMargs[constraint]
            lower, upper, need_lower, need_upper = self._process_M_value(m, lower, upper, need_lower, need_upper, bigMargs, constraint, constraint, from_args=True)
            if not need_lower and (not need_upper):
                return (lower, upper)
        elif parent in bigMargs:
            m = bigMargs[parent]
            lower, upper, need_lower, need_upper = self._process_M_value(m, lower, upper, need_lower, need_upper, bigMargs, parent, constraint, from_args=True)
            if not need_lower and (not need_upper):
                return (lower, upper)
        for arg in arg_list:
            for block, val in arg.items():
                lower, upper, need_lower, need_upper = self._process_M_value(val, lower, upper, need_lower, need_upper, bigMargs, block, constraint, from_args=True)
                if not need_lower and (not need_upper):
                    return (lower, upper)
        if None in bigMargs:
            m = bigMargs[None]
            lower, upper, need_lower, need_upper = self._process_M_value(m, lower, upper, need_lower, need_upper, bigMargs, None, constraint, from_args=True)
            if not need_lower and (not need_upper):
                return (lower, upper)
        return (lower, upper)

    def _estimate_M(self, expr, constraint):
        expr_lb, expr_ub = self._expr_bound_visitor.walk_expression(expr)
        if expr_lb == -interval.inf or expr_ub == interval.inf:
            raise GDP_Error("Cannot estimate M for unbounded expressions.\n\t(found while processing constraint '%s'). Please specify a value of M or ensure all variables that appear in the constraint are bounded." % constraint.name)
        else:
            M = (expr_lb, expr_ub)
        return tuple(M)

    def _add_constraint_expressions(self, c, i, M, indicator_var, newConstraint, constraintMap):
        unique = len(newConstraint)
        name = c.local_name + '_%s' % unique
        if c.lower is not None:
            if M[0] is None:
                raise GDP_Error("Cannot relax disjunctive constraint '%s' because M is not defined." % name)
            M_expr = M[0] * (1 - indicator_var)
            newConstraint.add((name, i, 'lb'), c.lower <= c.body - M_expr)
            constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'lb']]
            constraintMap['srcConstraints'][newConstraint[name, i, 'lb']] = c
        if c.upper is not None:
            if M[1] is None:
                raise GDP_Error("Cannot relax disjunctive constraint '%s' because M is not defined." % name)
            M_expr = M[1] * (1 - indicator_var)
            newConstraint.add((name, i, 'ub'), c.body - M_expr <= c.upper)
            transformed = constraintMap['transformedConstraints'].get(c)
            if transformed is not None:
                constraintMap['transformedConstraints'][c].append(newConstraint[name, i, 'ub'])
            else:
                constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'ub']]
            constraintMap['srcConstraints'][newConstraint[name, i, 'ub']] = c