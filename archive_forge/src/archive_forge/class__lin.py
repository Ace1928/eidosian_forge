from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
class _lin(object):
    """
    Vector valued linear function.


    Attributes:

    _coeff       dictionary {variable: coefficient}.  The coefficients
                 are dense or sparse matrices.  Scalar coefficients are
                 stored as 1x1 'd' matrices.


    Methods:

    value()        returns the value of the function: None if one of the
                   variables has value None;  a dense 'd' matrix of size
                   (len(self),1) if all the variables have values
    variables()    returns a (copy of) the list of variables
    _addterm()     adds a linear term  a*v
    _mul()         in-place multiplication
    _rmul()        in-place right multiplication
    """

    def __init__(self):
        self._coeff = {}

    def __len__(self):
        for v, c in iter(self._coeff.items()):
            if c.size[0] > 1:
                return c.size[0]
            elif _isscalar(c) and len(v) > 1:
                return len(v)
        return 1

    def __repr__(self):
        return '<linear function of length %d>' % len(self)

    def __str__(self):
        s = repr(self)[1:-1] + '\n'
        for v, c in iter(self._coeff.items()):
            s += 'coefficient of ' + repr(v) + ':\n' + str(c)
        return s

    def value(self):
        value = matrix(0.0, (len(self), 1))
        for v, c in iter(self._coeff.items()):
            if v.value is None:
                return None
            else:
                value += c * v.value
        return value

    def variables(self):
        return varlist(self._coeff.keys())

    def _addterm(self, a, v):
        """ 
        self += a*v  with v variable and a int, float, 1x1 dense 'd' 
        matrix, or sparse or dense 'd' matrix with len(v) columns.
        """
        lg = len(self)
        if v in self._coeff:
            c = self._coeff[v]
            if _ismatrix(a) and a.size[0] > 1 and (a.size[1] == len(v)) and (lg == 1 or lg == a.size[0]):
                newlg = a.size[0]
                if c.size == a.size:
                    self._coeff[v] = c + a
                elif c.size == (1, len(v)):
                    self._coeff[v] = c[newlg * [0], :] + a
                elif _isdmatrix(c) and c.size == (1, 1):
                    m = +a
                    m[::newlg + 1] += c[0]
                    self._coeff[v] = m
                else:
                    raise TypeError('incompatible dimensions')
            elif _ismatrix(a) and a.size == (1, len(v)):
                if c.size == (lg, len(v)):
                    self._coeff[v] = c + a[lg * [0], :]
                elif c.size == (1, len(v)):
                    self._coeff[v] = c + a
                elif _isdmatrix(c) and c.size == (1, 1):
                    m = a[lg * [0], :]
                    m[::lg + 1] += c[0]
                    self._coeff[v] = m
                else:
                    raise TypeError('incompatible dimensions')
            elif _isscalar(a) and len(v) > 1 and (lg == 1 or lg == len(v)):
                newlg = len(v)
                if c.size == (newlg, len(v)):
                    self._coeff[v][::newlg + 1] = c[::newlg + 1] + a
                elif c.size == (1, len(v)):
                    self._coeff[v] = c[newlg * [0], :]
                    self._coeff[v][::newlg + 1] = c[::newlg + 1] + a
                elif _isscalar(c):
                    self._coeff[v] = c + a
                else:
                    raise TypeError('incompatible dimensions')
            elif _isscalar(a) and len(v) == 1:
                self._coeff[v] = c + a
            else:
                raise TypeError('coefficient has invalid type or incompatible dimensions ')
        elif type(v) is variable:
            if _isscalar(a) and (lg == 1 or len(v) == 1 or len(v) == lg):
                self._coeff[v] = matrix(a, tc='d')
            elif _ismatrix(a) and a.size[1] == len(v) and (lg == 1 or a.size[0] == 1 or a.size[0] == lg):
                self._coeff[v] = +a
            else:
                raise TypeError('coefficient has invalid type or incompatible dimensions ')
        else:
            raise TypeError('second argument must be a variable')

    def _mul(self, a):
        """ 
        self := self*a where a is scalar or matrix 
        """
        if type(a) is int or type(a) is float:
            for v in iter(self._coeff.keys()):
                self._coeff[v] *= a
        elif _ismatrix(a) and a.size == (1, 1):
            for v in iter(self._coeff.keys()):
                self._coeff[v] *= a[0]
        elif len(self) == 1 and _isdmatrix(a) and (a.size[1] == 1):
            for v, c in iter(self._coeff.items()):
                self._coeff[v] = a * c
        else:
            raise TypeError('incompatible dimensions')

    def _rmul(self, a):
        """ 
        self := a*self where a is scalar or matrix 
        """
        lg = len(self)
        if _isscalar(a):
            for v in iter(self._coeff.keys()):
                self._coeff[v] *= a
        elif lg == 1 and _ismatrix(a) and (a.size[1] == 1):
            for v, c in iter(self._coeff.items()):
                self._coeff[v] = a * c
        elif _ismatrix(a) and a.size[1] == lg:
            for v, c in iter(self._coeff.items()):
                if c.size == (1, len(v)):
                    self._coeff[v] = a * c[lg * [0], :]
                else:
                    self._coeff[v] = a * c
        else:
            raise TypeError('incompatible dimensions')

    def __pos__(self):
        f = _lin()
        for v, c in iter(self._coeff.items()):
            f._coeff[v] = +c
        return f

    def __neg__(self):
        f = _lin()
        for v, c in iter(self._coeff.items()):
            f._coeff[v] = -c
        return f

    def __add__(self, other):
        f = +self
        if type(other) is int or (type(other) is float and (not other)):
            return f
        if type(other) is variable:
            f._addterm(1.0, other)
        elif type(other) is _lin:
            for v, c in iter(other._coeff.items()):
                f._addterm(c, v)
        else:
            return NotImplemented
        return f

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        """
        self += other  
        
        Only allowed if it does not change the length of self.
        """
        lg = len(self)
        if type(other) is variable and (len(other) == 1 or len(other) == lg):
            self._addterm(1.0, other)
        elif type(other) is _lin and (len(other) == 1 or len(other) == lg):
            for v, c in iter(other._coeff.items()):
                self._addterm(c, v)
        else:
            raise NotImplementedError('in-place addition must result in a function of the same length')
        return self

    def __sub__(self, other):
        f = +self
        if type(other) is variable:
            f._addterm(-1.0, other)
        elif type(other) is _lin:
            for v, c in iter(other._coeff.items()):
                f._addterm(-c, v)
        else:
            return NotImplemented
        return f

    def __rsub__(self, other):
        f = -self
        if type(other) is variable:
            f._addterm(1.0, other)
        elif type(other) is _lin:
            for v, c in iter(other._coeff.items()):
                f._addterm(c, v)
        else:
            return NotImplemented
        return f

    def __isub__(self, other):
        """
        self -= other  
        
        Only allowed if it does not change the length of self.
        """
        lg = len(self)
        if type(other) is variable and (len(other) == 1 or len(other) == lg):
            self._addterm(-1.0, other)
        elif type(other) is _lin and (len(other) == 1 or len(other) == lg):
            for v, c in iter(other._coeff.items()):
                self._addterm(-c, v)
        else:
            raise NotImplementedError('in-place subtraction must result in a function of the same length')
        return self

    def __mul__(self, other):
        if _isscalar(other) or _ismatrix(other):
            f = +self
            f._mul(other)
        else:
            return NotImplemented
        return f

    def __rmul__(self, other):
        if _isscalar(other) or _ismatrix(other):
            f = +self
            f._rmul(other)
        else:
            return NotImplemented
        return f

    def __imul__(self, other):
        """
        self *= other  
        
        Only allowed for scalar multiplication with a constant (int, 
        float, 1x1 'd' matrix).
        """
        if _isscalar(other):
            self._mul(other)
        else:
            raise NotImplementedError('in-place multiplication only defined for scalar multiplication')
        return self

    def __getitem__(self, key):
        l = _keytolist(key, len(self))
        if not l:
            raise ValueError('empty index set')
        f = _lin()
        for v, c in iter(self._coeff.items()):
            if c.size == (len(self), len(v)):
                f._coeff[v] = c[l, :]
            elif _isscalar(c) and len(v) == 1:
                f._coeff[v] = matrix(c, tc='d')
            elif c.size == (1, 1) and len(v) > 1:
                f._coeff[v] = spmatrix(c[0], range(len(l)), l, (len(l), len(v)), 'd')
            else:
                f._coeff[v] = c[len(l) * [0], :]
        return f