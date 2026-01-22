from io import StringIO
from pygments.formatter import Formatter
from pygments.lexer import Lexer, do_insertions
from pygments.token import Token, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt
def _filter_to(self, it, pred):
    """ Keep only the tokens that match `pred`, merge the others together """
    buf = ''
    idx = 0
    for i, t, v in it:
        if pred(t):
            if buf:
                yield (idx, None, buf)
                buf = ''
            yield (i, t, v)
        else:
            if not buf:
                idx = i
            buf += v
    if buf:
        yield (idx, None, buf)