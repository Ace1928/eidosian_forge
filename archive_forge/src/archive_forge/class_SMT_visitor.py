import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
class SMT_visitor(StreamBasedExpressionVisitor):
    """Creates an SMT expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    SMT string representation of an equivalent expression

    """

    def __init__(self, varmap):
        super(SMT_visitor, self).__init__()
        self.variable_label_map = varmap

    def exitNode(self, node, data):
        if isinstance(node, EqualityExpression):
            ans = '(= ' + data[0] + ' ' + data[1] + ')'
        elif isinstance(node, InequalityExpression):
            ans = '(<= ' + data[0] + ' ' + data[1] + ')'
        elif isinstance(node, ProductExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = '(* ' + ans + ' ' + arg + ')'
        elif isinstance(node, SumExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = '(+ ' + ans + ' ' + arg + ')'
        elif isinstance(node, PowExpression):
            ans = '(^ ' + data[0] + ' ' + data[1] + ')'
        elif isinstance(node, NegationExpression):
            ans = '(- 0 ' + data[0] + ')'
        elif isinstance(node, MonomialTermExpression):
            ans = '(* ' + data[0] + ' ' + data[1] + ')'
        elif isinstance(node, DivisionExpression):
            ans = '(/ ' + data[0] + ' ' + data[1] + ')'
        elif isinstance(node, AbsExpression):
            ans = '(abs ' + data[0] + ')'
        elif isinstance(node, UnaryFunctionExpression):
            if node.name == 'exp':
                ans = '(exp ' + data[0] + ')'
            elif node.name == 'log':
                raise NotImplementedError('logarithm not handled by z3 interface')
            elif node.name == 'sin':
                ans = '(sin ' + data[0] + ')'
            elif node.name == 'cos':
                ans = '(cos ' + data[0] + ')'
            elif node.name == 'tan':
                ans = '(tan ' + data[0] + ')'
            elif node.name == 'asin':
                ans = '(asin ' + data[0] + ')'
            elif node.name == 'acos':
                ans = '(acos ' + data[0] + ')'
            elif node.name == 'atan':
                ans = '(atan ' + data[0] + ')'
            elif node.name == 'sqrt':
                ans = '(^ ' + data[0] + ' (/ 1 2))'
            else:
                raise NotImplementedError('Unknown unary function: %s' % (node.name,))
        else:
            raise NotImplementedError(str(type(node)) + ' expression not handled by z3 interface')
        return ans

    def beforeChild(self, node, child, child_idx):
        if type(child) in nonpyomo_leaf_types:
            return (False, str(child))
        elif child.is_expression_type():
            return (True, '')
        elif child.is_numeric_type():
            if child.is_fixed():
                return (False, str(value(child)))
            else:
                return (False, str(self.variable_label_map.getSymbol(child)))
        else:
            return (False, str(child))

    def finalizeResult(self, node_result):
        return node_result