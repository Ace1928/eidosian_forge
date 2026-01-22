from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def _do_eval_formula_tests(tests):
    for code, result in six.iteritems(tests):
        if len(result) == 2:
            result = (False, []) + result
        model_desc = ModelDesc.from_formula(code)
        print(repr(code))
        print(result)
        print(model_desc)
        lhs_intercept, lhs_termlist, rhs_intercept, rhs_termlist = result
        _assert_terms_match(model_desc.lhs_termlist, lhs_intercept, lhs_termlist)
        _assert_terms_match(model_desc.rhs_termlist, rhs_intercept, rhs_termlist)