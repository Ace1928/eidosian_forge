from __future__ import print_function
import tokenize
import six
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter
def test_parse_origin():
    tree = parse_formula('a ~ b + c')
    assert tree.origin == Origin('a ~ b + c', 0, 9)
    assert tree.token.origin == Origin('a ~ b + c', 2, 3)
    assert tree.args[0].origin == Origin('a ~ b + c', 0, 1)
    assert tree.args[1].origin == Origin('a ~ b + c', 4, 9)
    assert tree.args[1].token.origin == Origin('a ~ b + c', 6, 7)
    assert tree.args[1].args[0].origin == Origin('a ~ b + c', 4, 5)
    assert tree.args[1].args[1].origin == Origin('a ~ b + c', 8, 9)