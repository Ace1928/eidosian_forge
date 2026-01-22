from __future__ import absolute_import
import types
from . import Errors
def CodeRange(code1, code2):
    """
    CodeRange(code1, code2) is an RE which matches any character
    with a code |c| in the range |code1| <= |c| < |code2|.
    """
    if code1 <= nl_code < code2:
        return Alt(RawCodeRange(code1, nl_code), RawNewline, RawCodeRange(nl_code + 1, code2))
    else:
        return RawCodeRange(code1, code2)