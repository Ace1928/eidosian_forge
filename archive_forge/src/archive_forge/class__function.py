from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
class _function(object):
    """
    Vector valued function.

    General form: 

        f = constant + linear + sum of nonlinear convex terms + 
            sum of nonlinear concave terms 

    The length of f is the maximum of the lengths of the terms in the 
    sum.  Each term must have length 1 or length equal to len(f).

    _function() creates the constant function f=0 with length 1.


    Attributes:

    _constant      constant term as a 1-column dense 'd' matrix of 
                   length 1 or length len(self)
    _linear        linear term as a _lin  object of length 1 or length
                   len(self)
    _cvxterms      nonlinear convex terms as a list [f1,f2,...] with 
                   each fi of type _minmax or _sum_minmax.  Each fi has
                   length 1 or length equal to len(self).
    _ccvterms      nonlinear concave terms as a list [f1,f2,...] with 
                   each fi of type _minmax or _sum_minmax.  Each fi has
                   length 1 or length equal to len(self).


    Methods:

    value()        returns the value of the function: None if one of the
                   variables has value None;  a dense 'd' matrix of size
                   (len(self),1) if all the variables have values
    variables()    returns a (copy of) the list of variables
    _iszero()      True if self is identically zero
    _isconstant()  True if there are no linear/convex/concave terms
    _islinear()    True if there are no constant/convex/concave terms
    _isaffine()    True if there are no nonlinear convex/concave terms 
    _isconvex()    True if there are no nonlinear concave terms
    _isconcave()   True if there are no nonlinear convex terms
    """

    def __init__(self):
        self._constant = matrix(0.0)
        self._linear = _lin()
        self._cvxterms = []
        self._ccvterms = []

    def __len__(self):
        if len(self._constant) > 1:
            return len(self._constant)
        lg = len(self._linear)
        if lg > 1:
            return lg
        for f in self._cvxterms:
            lg = len(f)
            if lg > 1:
                return lg
        for f in self._ccvterms:
            lg = len(f)
            if lg > 1:
                return lg
        return 1

    def __repr__(self):
        if self._iszero():
            return '<zero function of length %d>' % len(self)
        elif self._isconstant():
            return '<constant function of length %d>' % len(self)
        elif self._islinear():
            return '<linear function of length %d>' % len(self)
        elif self._isaffine():
            return '<affine function of length %d>' % len(self)
        elif self._isconvex():
            return '<convex function of length %d>' % len(self)
        elif self._isconcave():
            return '<concave function of length %d>' % len(self)
        else:
            return '<function of length %d>' % len(self)

    def __str__(self):
        s = repr(self)[1:-1]
        if not self._iszero() and (len(self._constant) != 1 or self._constant[0]):
            s += '\nconstant term:\n' + str(self._constant)
        else:
            s += '\n'
        if self._linear._coeff:
            s += 'linear term: ' + str(self._linear)
        if self._cvxterms:
            s += '%d nonlinear convex term(s):' % len(self._cvxterms)
            for f in self._cvxterms:
                s += '\n' + str(f)
        if self._ccvterms:
            s += '%d nonlinear concave term(s):' % len(self._ccvterms)
            for f in self._ccvterms:
                s += '\n' + str(f)
        return s

    def value(self):
        val = self._constant
        if self._linear._coeff:
            nval = self._linear.value()
            if nval is None:
                return None
            else:
                val = val + nval
        for f in self._cvxterms:
            nval = f.value()
            if nval is None:
                return None
            else:
                val = val + nval
        for f in self._ccvterms:
            nval = f.value()
            if nval is None:
                return None
            else:
                val = val + nval
        return val

    def variables(self):
        l = self._linear.variables()
        for f in self._cvxterms:
            l += [v for v in f.variables() if v not in l]
        for f in self._ccvterms:
            l += [v for v in f.variables() if v not in l]
        return l

    def _iszero(self):
        if not self._linear._coeff and (not self._cvxterms) and (not self._ccvterms) and (not blas.nrm2(self._constant)):
            return True
        else:
            return False

    def _isconstant(self):
        if not self._linear._coeff and (not self._cvxterms) and (not self._ccvterms):
            return True
        else:
            return False

    def _islinear(self):
        if len(self._constant) == 1 and (not self._constant[0]) and (not self._cvxterms) and (not self._ccvterms):
            return True
        else:
            return False

    def _isaffine(self):
        if not self._cvxterms and (not self._ccvterms):
            return True
        else:
            return False

    def _isconvex(self):
        if not self._ccvterms:
            return True
        else:
            return False

    def _isconcave(self):
        if not self._cvxterms:
            return True
        else:
            return False

    def __pos__(self):
        f = _function()
        f._constant = +self._constant
        f._linear = +self._linear
        f._cvxterms = [+g for g in self._cvxterms]
        f._ccvterms = [+g for g in self._ccvterms]
        return f

    def __neg__(self):
        f = _function()
        f._constant = -self._constant
        f._linear = -self._linear
        f._ccvterms = [-g for g in self._cvxterms]
        f._cvxterms = [-g for g in self._ccvterms]
        return f

    def __add__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):
            if other.size[1] != 1:
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented
        if 1 != len(self) != len(other) != 1:
            raise ValueError('incompatible lengths')
        f = _function()
        if _ismatrix(other):
            f._constant = self._constant + other
            f._linear = +self._linear
            f._cvxterms = [+fk for fk in self._cvxterms]
            f._ccvterms = [+fk for fk in self._ccvterms]
        else:
            if not (self._isconvex() and other._isconvex()) and (not (self._isconcave() and other._isconcave())):
                raise ValueError('operands must be both convex or both concave')
            f._constant = self._constant + other._constant
            f._linear = self._linear + other._linear
            f._cvxterms = [+fk for fk in self._cvxterms] + [+fk for fk in other._cvxterms]
            f._ccvterms = [+fk for fk in self._ccvterms] + [+fk for fk in other._ccvterms]
        return f

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):
            if other.size[1] != 1:
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented
        if len(self) != len(other) != 1:
            raise ValueError('incompatible lengths')
        if _ismatrix(other):
            if 1 == len(self._constant) != len(other):
                self._constant = self._constant + other
            else:
                self._constant += other
        else:
            if not (self._isconvex() and other._isconvex()) and (not (self._isconcave() and other._isconcave())):
                raise ValueError('operands must be both convex or both concave')
            if 1 == len(self._constant) != len(other._constant):
                self._constant = self._constant + other._constant
            else:
                self._constant += other._constant
            if 1 == len(self._linear) != len(other._linear):
                self._linear = self._linear + other._linear
            else:
                self._linear += other._linear
            self._cvxterms += [+fk for fk in other._cvxterms]
            self._ccvterms += [+fk for fk in other._ccvterms]
        return self

    def __sub__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):
            if other.size[1] != 1:
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented
        if 1 != len(self) != len(other) != 1:
            raise ValueError('incompatible lengths')
        f = _function()
        if _ismatrix(other):
            f._constant = self._constant - other
            f._linear = +self._linear
            f._cvxterms = [+fk for fk in self._cvxterms]
            f._ccvterms = [+fk for fk in self._ccvterms]
        else:
            if not (self._isconvex() and other._isconcave()) and (not (self._isconcave() and other._isconvex())):
                raise ValueError('operands must be convex and concave or concave and convex')
            f._constant = self._constant - other._constant
            f._linear = self._linear - other._linear
            f._cvxterms = [+fk for fk in self._cvxterms] + [-fk for fk in other._ccvterms]
            f._ccvterms = [+fk for fk in self._ccvterms] + [-fk for fk in other._cvxterms]
        return f

    def __rsub__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _isdmatrix(other):
            if other.size[1] != 1:
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented
        if 1 != len(self) != len(other) != 1:
            raise ValueError('incompatible lengths')
        f = _function()
        if _ismatrix(other):
            f._constant = other - self._constant
            f._linear = -self._linear
            f._cvxterms = [-fk for fk in self._ccvterms]
            f._ccvterms = [-fk for fk in self._cvxterms]
        else:
            if not (self._isconvex() and other._isconcave()) and (not (self._isconcave() and other._isconvex())):
                raise ValueError('operands must be convex and concave or concave and convex')
            f._constant = other._constant - self._constant
            f._linear = other._linear - self._linear
            f._cvxterms = [-fk for fk in self._ccvterms] + [fk for fk in other._cvxterms]
            f._ccvterms = [-fk for fk in self._cvxterms] + [fk for fk in other._ccvterms]
        return f

    def __isub__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):
            if other.size[1] != 1:
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented
        if len(self) != len(other) != 1:
            raise ValueError('incompatible lengths')
        if _ismatrix(other):
            if 1 == len(self._constant) != len(other):
                self._constant = self._constant - other
            else:
                self._constant -= other
        else:
            if not (self._isconvex() and other._isconcave()) and (not (self._isconcave() and other._isconvex())):
                raise ValueError('operands must be convex and concave or concave and convex')
            if 1 == len(self._constant) != len(other._constant):
                self._constant = self._constant - other._constant
            else:
                self._constant -= other._constant
            if 1 == len(self._linear) != len(other._linear):
                self._linear = self._linear - other._linear
            else:
                self._linear -= other._linear
            self._cvxterms += [-fk for fk in other._ccvterms]
            self._ccvterms += [-fk for fk in other._cvxterms]
        return self

    def __mul__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        if _ismatrix(other) and other.size == (1, 1) or (_isdmatrix(other) and other.size[1] == 1 == len(self)):
            f = _function()
            if other.size == (1, 1) and other[0] == 0.0:
                f._constant = matrix(0.0, (len(self), 1))
                return f
            if len(self._constant) != 1 or self._constant[0]:
                f._constant = self._constant * other
            if self._linear._coeff:
                f._linear = self._linear * other
            if not self._isaffine():
                if other.size == (1, 1):
                    if other[0] > 0.0:
                        f._cvxterms = [fk * other[0] for fk in self._cvxterms]
                        f._ccvterms = [fk * other[0] for fk in self._ccvterms]
                    elif other[0] < 0.0:
                        f._cvxterms = [fk * other[0] for fk in self._ccvterms]
                        f._ccvterms = [fk * other[0] for fk in self._cvxterms]
                    else:
                        pass
                else:
                    raise ValueError('can only multiply with scalar')
        else:
            raise TypeError('incompatible dimensions or types')
        return f

    def __rmul__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        lg = len(self)
        if _ismatrix(other) and other.size[1] == lg or (_isdmatrix(other) and other.size == (1, 1)):
            f = _function()
            if other.size == (1, 1) and other[0] == 0.0:
                f._constant = matrix(0.0, (len(self), 1))
                return f
            if len(self._constant) != 1 or self._constant[0]:
                if 1 == len(self._constant) != lg and (not _isscalar(other)):
                    f._constant = other * self._constant[lg * [0]]
                else:
                    f._constant = other * self._constant
            if self._linear._coeff:
                if 1 == len(self._linear) != lg and (not _isscalar(other)):
                    f._linear = other * self._linear[lg * [0]]
                else:
                    f._linear = other * self._linear
            if not self._isaffine():
                if other.size == (1, 1):
                    if other[0] > 0.0:
                        f._cvxterms = [other[0] * fk for fk in self._cvxterms]
                        f._ccvterms = [other[0] * fk for fk in self._ccvterms]
                    elif other[0] < 0.0:
                        f._cvxterms = [other[0] * fk for fk in self._ccvterms]
                        f._ccvterms = [other[0] * fk for fk in self._cvxterms]
                    else:
                        pass
                else:
                    raise ValueError('can only multiply with scalar')
        else:
            raise TypeError('incompatible dimensions or types')
        return f

    def __imul__(self, other):
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        if _isdmatrix(other) and other.size == (1, 1):
            if other[0] == 0.0:
                self._constant = matrix(0.0, (len(self), 1))
                return self
            if len(self._constant) != 1 or self._constant[0]:
                self._constant *= other[0]
            if self._linear._coeff:
                self._linear *= other[0]
            if not self._isaffine():
                if other[0] > 0.0:
                    for f in self._cvxterms:
                        f *= other[0]
                    for f in self._ccvterms:
                        f *= other[0]
                elif other[0] < 0.0:
                    cvxterms = [f * other[0] for f in self._ccvterms]
                    self._ccvterms = [f * other[0] for f in self._cvxterms]
                    self._cvxterms = cvxterms
                else:
                    pass
            return self
        else:
            raise TypeError('incompatible dimensions or types')
    if sys.version_info[0] < 3:

        def __div__(self, other):
            if type(other) is int or type(other) is float:
                return self.__mul__(1.0 / other)
            elif _isdmatrix(other) and other.size == (1, 1):
                return self.__mul__(1.0 / other[0])
            else:
                return NotImplemented
    else:

        def __truediv__(self, other):
            if type(other) is int or type(other) is float:
                return self.__mul__(1.0 / other)
            elif _isdmatrix(other) and other.size == (1, 1):
                return self.__mul__(1.0 / other[0])
            else:
                return NotImplemented

    def __rdiv__(self, other):
        return NotImplemented
    if sys.version_info[0] < 3:

        def __idiv__(self, other):
            if type(other) is int or type(other) is float:
                return self.__imul__(1.0 / other)
            elif _isdmatrix(other) and other.size == (1, 1):
                return self.__imul__(1.0 / other[0])
            else:
                return NotImplemented
    else:

        def __itruediv__(self, other):
            if type(other) is int or type(other) is float:
                return self.__imul__(1.0 / other)
            elif _isdmatrix(other) and other.size == (1, 1):
                return self.__imul__(1.0 / other[0])
            else:
                return NotImplemented

    def __abs__(self):
        return max(self, -self)

    def __eq__(self, other):
        return constraint(self - other, '=')

    def __le__(self, other):
        return constraint(self - other, '<')

    def __ge__(self, other):
        return constraint(other - self, '<')

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
        lg = len(self)
        l = _keytolist(key, lg)
        if not l:
            raise ValueError('empty index set')
        f = _function()
        if len(self._constant) != 1 or self._constant[0]:
            if 1 == len(self._constant) != lg:
                f._constant = +self._constant
            else:
                f._constant = self._constant[l]
        if self._linear:
            if 1 == len(self._linear) != lg:
                f._linear = +self._linear
            else:
                f._linear = self._linear[l]
        for fk in self._cvxterms:
            if 1 == len(fk) != lg:
                f._cvxterms += [+fk]
            elif type(fk) is _minmax:
                f._cvxterms += [fk[l]]
            else:
                fmax = _minmax('max', *fk._flist)
                f._cvxterms += [gk[l] for gk in fmax]
        for fk in self._ccvterms:
            if 1 == len(fk) != lg:
                f._ccvterms += [+fk]
            elif type(fk) is _minmax:
                f._ccvterms += [fk[l]]
            else:
                fmin = _minmax('min', *fk._flist)
                f._ccvterms += [gk[l] for gk in fmin]
        return f