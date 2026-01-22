from __future__ import print_function
import tokenize
import six
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter
def _tokenize_formula(code, operator_strings):
    assert '(' not in operator_strings
    assert ')' not in operator_strings
    magic_token_types = {'(': Token.LPAREN, ')': Token.RPAREN}
    for operator_string in operator_strings:
        magic_token_types[operator_string] = operator_string
    end_tokens = set(magic_token_types)
    end_tokens.remove('(')
    it = PushbackAdapter(python_tokenize(code))
    for pytype, token_string, origin in it:
        if token_string in magic_token_types:
            yield Token(magic_token_types[token_string], origin)
        else:
            it.push_back((pytype, token_string, origin))
            yield _read_python_expr(it, end_tokens)