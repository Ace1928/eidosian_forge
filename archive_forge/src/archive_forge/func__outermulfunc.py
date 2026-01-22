import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _outermulfunc(a, b):
    if _isvec(a) and _isvec(b):
        return '%svOuterProduct(%s,%s)' % (mprefix, a[vplen:], b[vplen:])
    else:
        raise TypeError