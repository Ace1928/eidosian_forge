from __future__ import absolute_import
import types
from . import Errors
def Str1(s):
    """
    Str1(s) is an RE which matches the literal string |s|.
    """
    result = Seq(*tuple(map(Char, s)))
    result.str = 'Str(%s)' % repr(s)
    return result