from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
class _sum_minmax(_minmax):
    """
    Sum of componentwise maximum or minimum of functions.  

    A function of the form f = sum(max(f1,f2,...,fm)) or 
    f = sum(min(f1,f2,...,fm)) with each fi an object of 
    type _function.  

    m must be greater than 1.  len(f) = 1.
    Each fi has length 1 or length equal to max_i len(fi)).


    Attributes:

    _flist       [f1,f2,...,fm]
    _ismax       True for 'max', False for 'min'


    Methods:

    value()      returns the value of the function
    variables()  returns a copy of the list of variables
    _length()    number of terms in the sum
    """

    def __init__(self, op, *s):
        _minmax.__init__(self, op, *s)
        if len(self._flist) == 1:
            raise TypeError('expected more than 1 argument')

    def __len__(self):
        return 1

    def _length(self):
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
        return '<sum of componentwise ' + s + ' of %d functions of length %d>' % (len(self._flist), len(self))

    def __str__(self):
        s = repr(self)[1:-1]
        for k in range(len(self._flist)):
            s += '\nfunction %d: ' % k + repr(self._flist[k])[1:-1]
        return s

    def value(self):
        if self._ismax:
            return matrix(sum(_vecmax(*[f.value() for f in self._flist])), tc='d')
        else:
            return matrix(sum(_vecmin(*[f.value() for f in self._flist])), tc='d')

    def __pos__(self):
        if self._ismax:
            f = _sum_minmax('max', *[+fk for fk in self._flist])
        else:
            f = _sum_minmax('min', *[+fk for fk in self._flist])
        return f

    def __neg__(self):
        if self._ismax:
            f = _sum_minmax('min', *[-fk for fk in self._flist])
        else:
            f = _sum_minmax('max', *[-fk for fk in self._flist])
        return f

    def __mul__(self, other):
        if type(other) is int or type(other) is float or (_ismatrix(other) and other.size == (1, 1)):
            if _ismatrix(other):
                other = other[0]
            if other >= 0.0:
                if self._ismax:
                    f = _sum_minmax('max', *[other * fk for fk in self._flist])
                else:
                    f = _sum_minmax('min', *[other * fk for fk in self._flist])
            elif self._ismax:
                f = _sum_minmax('min', *[other * fk for fk in self._flist])
            else:
                f = _sum_minmax('max', *[other * fk for fk in self._flist])
            return f
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, key):
        l = _keytolist(key, 1)
        if not l:
            raise ValueError('empty index set')
        if self._ismax:
            f = sum(_minmax('max', *self._flist))
        else:
            f = sum(_minmax('min', *self._flist))
        return f[l]