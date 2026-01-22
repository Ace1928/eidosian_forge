from __future__ import absolute_import
import types
from . import Errors
def Char(c):
    """
    Char(c) is an RE which matches the character |c|.
    """
    if len(c) == 1:
        result = CodeRange(ord(c), ord(c) + 1)
    else:
        result = SpecialSymbol(c)
    result.str = 'Char(%s)' % repr(c)
    return result