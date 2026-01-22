from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
class IntermediateExpr(object):
    """This class holds an intermediate result while we're evaluating a tree."""

    def __init__(self, intercept, intercept_origin, intercept_removed, terms):
        self.intercept = intercept
        self.intercept_origin = intercept_origin
        self.intercept_removed = intercept_removed
        self.terms = tuple(uniqueify_list(terms))
        if self.intercept:
            assert self.intercept_origin
        assert not (self.intercept and self.intercept_removed)
    __repr__ = repr_pretty_delegate

    def _pretty_repr_(self, p, cycle):
        assert not cycle
        return repr_pretty_impl(p, self, [self.intercept, self.intercept_origin, self.intercept_removed, self.terms])
    __getstate__ = no_pickling