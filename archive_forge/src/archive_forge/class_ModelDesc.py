from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
class ModelDesc(object):
    """A simple container representing the termlists parsed from a formula.

    This is a simple container object which has exactly the same
    representational power as a formula string, but is a Python object
    instead. You can construct one by hand, and pass it to functions like
    :func:`dmatrix` or :func:`incr_dbuilder` that are expecting a formula
    string, but without having to do any messy string manipulation. For
    details see :ref:`expert-model-specification`.

    Attributes:

    .. attribute:: lhs_termlist
                   rhs_termlist

       Two termlists representing the left- and right-hand sides of a
       formula, suitable for passing to :func:`design_matrix_builders`.
    """

    def __init__(self, lhs_termlist, rhs_termlist):
        self.lhs_termlist = uniqueify_list(lhs_termlist)
        self.rhs_termlist = uniqueify_list(rhs_termlist)
    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        assert not cycle
        return repr_pretty_impl(p, self, [], [('lhs_termlist', self.lhs_termlist), ('rhs_termlist', self.rhs_termlist)])

    def describe(self):
        """Returns a human-readable representation of this :class:`ModelDesc`
        in pseudo-formula notation.

        .. warning:: There is no guarantee that the strings returned by this
           function can be parsed as formulas. They are best-effort
           descriptions intended for human users. However, if this ModelDesc
           was created by parsing a formula, then it should work in
           practice. If you *really* have to.
        """

        def term_code(term):
            if term == INTERCEPT:
                return '1'
            else:
                return term.name()
        result = ' + '.join([term_code(term) for term in self.lhs_termlist])
        if result:
            result += ' ~ '
        else:
            result += '~ '
        if self.rhs_termlist == [INTERCEPT]:
            result += term_code(INTERCEPT)
        else:
            term_names = []
            if INTERCEPT not in self.rhs_termlist:
                term_names.append('0')
            term_names += [term_code(term) for term in self.rhs_termlist if term != INTERCEPT]
            result += ' + '.join(term_names)
        return result

    @classmethod
    def from_formula(cls, tree_or_string):
        """Construct a :class:`ModelDesc` from a formula string.

        :arg tree_or_string: A formula string. (Or an unevaluated formula
          parse tree, but the API for generating those isn't public yet. Shh,
          it can be our secret.)
        :returns: A new :class:`ModelDesc`.
        """
        if isinstance(tree_or_string, ParseNode):
            tree = tree_or_string
        else:
            tree = parse_formula(tree_or_string)
        value = Evaluator().eval(tree, require_evalexpr=False)
        assert isinstance(value, cls)
        return value
    __getstate__ = no_pickling