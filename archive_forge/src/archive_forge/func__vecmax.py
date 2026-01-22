from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _vecmax(*s):
    """
    _vecmax(s1,s2,...) returns the componentwise maximum of s1, s2,... 
    _vecmax(s) returns the maximum component of s.
    
    The arguments can be None, scalars or 1-column dense 'd' vectors 
    with lengths equal to 1 or equal to the maximum len(sk).  

    Returns None if one of the arguments is None.  
    """
    if not s:
        raise TypeError('_vecmax expected at least 1 argument, got 0')
    val = None
    for c in s:
        if c is None:
            return None
        elif type(c) is int or type(c) is float:
            c = matrix(c, tc='d')
        elif not _isdmatrix(c) or c.size[1] != 1:
            raise TypeError('incompatible type or size')
        if val is None:
            if len(s) == 1:
                return matrix(max(c), tc='d')
            else:
                val = +c
        elif len(val) == 1 != len(c):
            val = matrix([max(val[0], x) for x in c], tc='d')
        elif len(val) != 1 == len(c):
            val = matrix([max(c[0], x) for x in val], tc='d')
        elif len(val) == len(c):
            val = matrix([max(val[k], c[k]) for k in range(len(c))], tc='d')
        else:
            raise ValueError('incompatible dimensions')
    return val