from __future__ import absolute_import
import types
from . import Errors
def uppercase_range(code1, code2):
    """
    If the range of characters from code1 to code2-1 includes any
    lower case letters, return the corresponding upper case range.
    """
    code3 = max(code1, ord('a'))
    code4 = min(code2, ord('z') + 1)
    if code3 < code4:
        d = ord('A') - ord('a')
        return (code3 + d, code4 + d)
    else:
        return None