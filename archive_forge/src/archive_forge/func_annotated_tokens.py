import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
def annotated_tokens(code):
    prev_was_dot = False
    it = PushbackAdapter(python_tokenize(code))
    for token_type, token, origin in it:
        props = {}
        props['bare_ref'] = not prev_was_dot and token_type == tokenize.NAME
        props['bare_funcall'] = props['bare_ref'] and it.has_more() and (it.peek()[1] == '(')
        yield (token_type, token, origin, props)
        prev_was_dot = token == '.'