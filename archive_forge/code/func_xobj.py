import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def xobj(symb, length):
    """Construct spatial object of given length.

    return: [] of equal-length strings
    """
    if length <= 0:
        raise ValueError('Length should be greater than 0')
    if _use_unicode:
        _xobj = _xobj_unicode
    else:
        _xobj = _xobj_ascii
    vinfo = _xobj[symb]
    c1 = top = bot = mid = None
    if not isinstance(vinfo, tuple):
        ext = vinfo
    else:
        if isinstance(vinfo[0], tuple):
            vlong = vinfo[0]
            c1 = vinfo[1]
        else:
            vlong = vinfo
        ext = vlong[0]
        try:
            top = vlong[1]
            bot = vlong[2]
            mid = vlong[3]
        except IndexError:
            pass
    if c1 is None:
        c1 = ext
    if top is None:
        top = ext
    if bot is None:
        bot = ext
    if mid is not None:
        if length % 2 == 0:
            length += 1
    else:
        mid = ext
    if length == 1:
        return c1
    res = []
    next = (length - 2) // 2
    nmid = length - 2 - next * 2
    res += [top]
    res += [ext] * next
    res += [mid] * nmid
    res += [ext] * next
    res += [bot]
    return res