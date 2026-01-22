import ctypes
import logging
import os
from pyomo.common.fileutils import Library
from pyomo.core import value, Expression
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.common.collections import ComponentMap
class MCPP_visitor(StreamBasedExpressionVisitor):
    """Creates an MC++ expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    expression of type McCormick.

    Note on memory management: The MCPP_visitor will return a pointer to
    an MC++ interval object that was dynamically allocated within the C
    interface.  It is the caller's responsibility to call
    `mcpp_lib.release()` on that object to prevent a memory leak

    """

    def __init__(self, expression, improved_var_bounds=None):
        super(MCPP_visitor, self).__init__()
        self.mcpp = _MCPP_lib()
        so_file_version = self.mcpp.get_version()
        so_file_version = so_file_version.decode('utf-8')
        if not so_file_version == __version__:
            raise MCPP_Error('Shared object file version %s is out of date with MC++ interface version %s. Please rebuild the library.' % (so_file_version, __version__))
        self.missing_value_warnings = []
        self.expr = expression
        vars = list(identify_variables(expression, include_fixed=False))
        self.num_vars = len(vars)
        self.known_vars = ComponentMap()
        self.var_to_idx = ComponentMap()
        inf = float('inf')
        for i, var in enumerate(vars):
            self.var_to_idx[var] = i
            if improved_var_bounds is not None:
                lb, ub = improved_var_bounds.get(var, (-inf, inf))
            else:
                lb, ub = (-inf, inf)
            self.known_vars[var] = self.register_var(var, lb, ub)
        self.refs = None

    def walk_expression(self):
        self.refs = set()
        ans = super(MCPP_visitor, self).walk_expression(self.expr)
        self.refs = None
        return ans

    def exitNode(self, node, data):
        if isinstance(node, ProductExpression):
            ans = self.mcpp.multiply(data[0], data[1])
        elif isinstance(node, SumExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = self.mcpp.add(ans, arg)
        elif isinstance(node, PowExpression):
            if type(node.arg(1)) == int:
                ans = self.mcpp.try_binary_fcn(self.mcpp.power, data[0], data[1])
            elif type(node.arg(1)) == float:
                ans = self.mcpp.try_binary_fcn(self.mcpp.powerf, data[0], data[1])
            else:
                ans = self.mcpp.try_binary_fcn(self.mcpp.powerx, data[0], data[1])
        elif isinstance(node, DivisionExpression):
            ans = self.mcpp.try_binary_fcn(self.mcpp.divide, data[0], data[1])
        elif isinstance(node, NegationExpression):
            ans = self.mcpp.negation(data[0])
        elif isinstance(node, AbsExpression):
            ans = self.mcpp.try_unary_fcn(self.mcpp.mc_abs, data[0])
        elif isinstance(node, LinearExpression):
            raise NotImplementedError('Quicksum has bugs that prevent proper usage of MC++.')
        elif isinstance(node, UnaryFunctionExpression):
            if node.name == 'exp':
                ans = self.mcpp.try_unary_fcn(self.mcpp.exponential, data[0])
            elif node.name == 'log':
                ans = self.mcpp.try_unary_fcn(self.mcpp.logarithm, data[0])
            elif node.name == 'sin':
                ans = self.mcpp.try_unary_fcn(self.mcpp.trigSin, data[0])
            elif node.name == 'cos':
                ans = self.mcpp.try_unary_fcn(self.mcpp.trigCos, data[0])
            elif node.name == 'tan':
                ans = self.mcpp.try_unary_fcn(self.mcpp.trigTan, data[0])
            elif node.name == 'asin':
                ans = self.mcpp.try_unary_fcn(self.mcpp.atrigSin, data[0])
            elif node.name == 'acos':
                ans = self.mcpp.try_unary_fcn(self.mcpp.atrigCos, data[0])
            elif node.name == 'atan':
                ans = self.mcpp.try_unary_fcn(self.mcpp.atrigTan, data[0])
            elif node.name == 'sqrt':
                ans = self.mcpp.try_unary_fcn(self.mcpp.mc_sqrt, data[0])
            else:
                raise NotImplementedError('Unknown unary function: %s' % (node.name,))
        elif isinstance(node, NPV_expressions):
            ans = self.mcpp.newConstant(value(data[0]))
        elif type(node) in nonpyomo_leaf_types:
            ans = self.mcpp.newConstant(node)
        elif not node.is_expression_type():
            ans = self.register_num(node)
        elif type(node) in SubclassOf(Expression) or isinstance(node, _ExpressionData):
            ans = data[0]
        else:
            raise RuntimeError('Unhandled expression type: %s' % type(node))
        if ans is None:
            msg = self.mcpp.get_last_exception_message()
            msg = msg.decode('utf-8')
            raise MCPP_Error(msg)
        return ans

    def beforeChild(self, node, child, child_idx):
        if type(child) in nonpyomo_leaf_types:
            return (False, self.mcpp.newConstant(child))
        elif not child.is_expression_type():
            return (False, self.register_num(child))
        else:
            return (True, None)

    def acceptChildResult(self, node, data, child_result, child_idx):
        self.refs.add(child_result)
        data.append(child_result)
        return data

    def register_num(self, num):
        """Registers a new number: Param, Var, or NumericConstant."""
        if num.is_fixed():
            return self.mcpp.newConstant(value(num))
        else:
            return self.known_vars[num]

    def register_var(self, var, lb, ub):
        """Registers a new variable."""
        var_idx = self.var_to_idx[var]
        inf = float('inf')
        lb = -inf if lb is None else lb
        ub = inf if ub is None else ub
        lb = max(var.lb if var.has_lb() else -inf, lb)
        ub = min(var.ub if var.has_ub() else inf, ub)
        var_val = value(var, exception=False)
        if lb == -inf:
            lb = -500000
            logger.warning('Var %s missing lower bound. Assuming LB of %s' % (var.name, lb))
        if ub == inf:
            ub = 500000
            logger.warning('Var %s missing upper bound. Assuming UB of %s' % (var.name, ub))
        if var_val is None:
            var_val = (lb + ub) / 2
            self.missing_value_warnings.append('Var %s missing value. Assuming midpoint value of %s' % (var.name, var_val))
        return self.mcpp.newVar(lb, var_val, ub, self.num_vars, var_idx)

    def finalizeResult(self, node_result):
        assert node_result not in self.refs
        for r in self.refs:
            self.mcpp.release(r)
        return node_result