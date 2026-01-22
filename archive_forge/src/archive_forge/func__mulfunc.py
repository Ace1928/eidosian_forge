import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _mulfunc(a, b):
    if _isscalar(a) and _isscalar(b):
        return '%s*%s' % (a, b)
    if _isvec(a) and _isvec(b):
        return 'vDot(%s,%s)' % (a[vplen:], b[vplen:])
    if _ismat(a) and _ismat(b):
        return '%smMultiply(%s,%s)' % (mprefix, a[mplen:], b[mplen:])
    if _ismat(a) and _isvec(b):
        return '%smvMultiply(%s,%s)' % (vprefix, a[mplen:], b[vplen:])
    if _ismat(a) and _isscalar(b):
        return '%smScale(%s,%s)' % (mprefix, a[mplen:], b)
    if _isvec(a) and _isscalar(b):
        return '%svScale(%s,%s)' % (vprefix, a[mplen:], b)
    else:
        raise TypeError