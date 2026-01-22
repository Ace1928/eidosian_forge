import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _expfunc(a, b):
    if _isscalar(a) and _isscalar(b):
        return 'pow(%s,%s)' % (str(a), str(b))
    if _ismat(a) and b == '-1':
        return '%smInverse(%s)' % (mprefix, a[mplen:])
    if _ismat(a) and b == 'T':
        return '%smTranspose(%s)' % (mprefix, a[mplen:])
    if _ismat(a) and b == 'Det':
        return 'mDeterminant(%s)' % a[mplen:]
    if _isvec(a) and b == 'Mag':
        return 'sqrt(vMagnitude2(%s))' % a[vplen:]
    if _isvec(a) and b == 'Mag2':
        return 'vMagnitude2(%s)' % a[vplen:]
    else:
        raise TypeError