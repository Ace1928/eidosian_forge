from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def _eval_binary_minus(evaluator, tree):
    left_expr = evaluator.eval(tree.args[0])
    if tree.args[1].type == 'ZERO':
        return IntermediateExpr(True, tree.args[1], False, left_expr.terms)
    elif tree.args[1].type == 'ONE':
        return IntermediateExpr(False, None, True, left_expr.terms)
    else:
        right_expr = evaluator.eval(tree.args[1])
        terms = [term for term in left_expr.terms if term not in right_expr.terms]
        if right_expr.intercept:
            return IntermediateExpr(False, None, True, terms)
        else:
            return IntermediateExpr(left_expr.intercept, left_expr.intercept_origin, left_expr.intercept_removed, terms)