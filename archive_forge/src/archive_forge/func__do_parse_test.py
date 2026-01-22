from __future__ import print_function
import tokenize
import six
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter
def _do_parse_test(test_cases, extra_operators):
    for code, expected in six.iteritems(test_cases):
        actual = parse_formula(code, extra_operators=extra_operators)
        print(repr(code), repr(expected))
        print(actual)
        _compare_trees(actual, expected)