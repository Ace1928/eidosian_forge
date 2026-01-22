from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
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