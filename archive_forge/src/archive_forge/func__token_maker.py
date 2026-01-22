from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def _token_maker(type, string):

    def make_token(scanner, token_string):
        if type == '__OP__':
            actual_type = token_string
        else:
            actual_type = type
        return Token(actual_type, Origin(string, *scanner.match.span()), token_string)
    return make_token