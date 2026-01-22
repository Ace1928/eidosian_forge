from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def _eval_binary_plus(evaluator, tree):
    left_expr = evaluator.eval(tree.args[0])
    if tree.args[1].type == 'ZERO':
        return IntermediateExpr(False, None, True, left_expr.terms)
    else:
        right_expr = evaluator.eval(tree.args[1])
        if right_expr.intercept:
            return IntermediateExpr(True, right_expr.intercept_origin, False, left_expr.terms + right_expr.terms)
        else:
            return IntermediateExpr(left_expr.intercept, left_expr.intercept_origin, left_expr.intercept_removed, left_expr.terms + right_expr.terms)