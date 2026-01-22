from __future__ import print_function
import tokenize
import six
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter
def _parsing_error_test(parse_fn, error_descs):
    for error_desc in error_descs:
        letters = []
        start = None
        end = None
        for letter in error_desc:
            if letter == '<':
                start = len(letters)
            elif letter == '>':
                end = len(letters)
            else:
                letters.append(letter)
        bad_code = ''.join(letters)
        assert start is not None and end is not None
        print(error_desc)
        print(repr(bad_code), start, end)
        try:
            parse_fn(bad_code)
        except PatsyError as e:
            print(e)
            assert e.origin.code == bad_code
            assert e.origin.start in (0, start)
            assert e.origin.end in (end, len(bad_code))
        else:
            assert False, 'parser failed to report an error!'