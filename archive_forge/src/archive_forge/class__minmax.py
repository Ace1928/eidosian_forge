from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
class _minmax(object):
    """
    Componentwise maximum or minimum of functions.  

    A function of the form f = max(f1,f2,...,fm) or f = max(f1) or
    f = min(f1,f2,...,fm) or f = min(f1) with each fi an object of 
    type _function.  

    If m>1, then len(f) = max(len(fi)) and f is the componentwise 
    maximum/minimum of f1,f2,...,fm.  Each fi has length 1 or length 
    equal to len(f).

    If m=1, then len(f) = 1 and f is the maximum/minimum of the 
    components of f1: f = max(f1[0],f1[1],...) or 
    f = min(f1[0],f1[1],...).
   

    Attributes:

    _flist       [f1,f2,...,fm]
    _ismax       True for 'max', False for 'min'


    Methods:

    value()      returns the value of the function
    variables()  returns a copy of the list of variables
    """

    def __init__(self, op, *s):
        self._flist = []
        if op == 'max':
            self._ismax = True
        else:
            self._ismax = False
        if len(s) == 1:
            if type(s[0]) is variable or (type(s[0]) is _function and (s[0]._isconvex() and self._ismax) or (s[0]._isconcave() and (not self._ismax))):
                self._flist += [+s[0]]
            else:
                raise TypeError('unsupported argument type')
        else:
            cnst = None
            lg = 1
            for f in s:
                if type(f) is int or type(f) is float:
                    f = matrix(f, tc='d')
                if _isdmatrix(f) and f.size[1] == 1:
                    if cnst is None:
                        cnst = +f
                    elif self._ismax:
                        cnst = _vecmax(cnst, f)
                    else:
                        cnst = _vecmin(cnst, f)
                elif type(f) is variable or type(f) is _function:
                    self._flist += [+f]
                else:
                    raise TypeError('unsupported argument type')
                lgf = len(f)
                if 1 != lg != lgf != 1:
                    raise ValueError('incompatible dimensions')
                elif 1 == lg != lgf:
                    lg = lgf
            if cnst is not None:
                self._flist += [_function() + cnst]

    def __len__(self):
        if len(self._flist) == 1:
            return 1
        for f in self._flist:
            lg = len(f)
            if len(f) > 1:
                return lg
        return 1

    def __repr__(self):
        if self._ismax:
            s = 'maximum'
        else:
            s = 'minimum'
        if len(self._flist) == 1:
            return '<' + s + ' component of a function of length %d>' % len(self._flist[0])
        else:
            return '<componentwise ' + s + ' of %d functions of length %d>' % (len(self._flist), len(self))

    def __str__(self):
        s = repr(self)[1:-1] + ':'
        if len(self._flist) == 1:
            s += '\n' + repr(self._flist[0])[1:-1]
        else:
            for k in range(len(self._flist)):
                s += '\nfunction %d: ' % k + repr(self._flist[k])[1:-1]
        return s

    def value(self):
        if self._ismax:
            return _vecmax(*[f.value() for f in self._flist])
        else:
            return _vecmin(*[f.value() for f in self._flist])

    def variables(self):
        l = varlist()
        for f in self._flist:
            l += [v for v in f.variables() if v not in l]
        return l

    def __pos__(self):
        if self._ismax:
            f = _minmax('max', *[+fk for fk in self._flist])
        else:
            f = _minmax('min', *[+fk for fk in self._flist])
        return f

    def __neg__(self):
        if self._ismax:
            f = _minmax('min', *[-fk for fk in self._flist])
        else:
            f = _minmax('max', *[-fk for fk in self._flist])
        return f

    def __mul__(self, other):
        if type(other) is int or type(other) is float or (_ismatrix(other) and other.size == (1, 1)):
            if _ismatrix(other):
                other = other[0]
            if other >= 0.0:
                if self._ismax:
                    f = _minmax('max', *[other * fk for fk in self._flist])
                else:
                    f = _minmax('min', *[other * fk for fk in self._flist])
            elif self._ismax:
                f = _minmax('min', *[other * fk for fk in self._flist])
            else:
                f = _minmax('max', *[other * fk for fk in self._flist])
            return f
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if _isscalar(other):
            if type(other) is matrix:
                other = other[0]
            for f in self._flist:
                f *= other
            if other < 0.0:
                self._ismax = not self._ismax
            return self
        raise NotImplementedError('in-place multiplication is only defined for scalars')

    def __getitem__(self, key):
        lg = len(self)
        l = _keytolist(key, lg)
        if not l:
            raise ValueError('empty index set')
        if len(self._flist) == 1:
            fl = list(self._flist[0])
        else:
            fl = self._flist
        if self._ismax:
            f = _minmax('max')
        else:
            f = _minmax('min')
        for fk in fl:
            if 1 == len(fk) != lg:
                f._flist += [+fk]
            else:
                f._flist += [fk[l]]
        return f