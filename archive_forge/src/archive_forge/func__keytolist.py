from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _keytolist(key, n):
    """
    Converts indices, index lists, index matrices, and slices of
    a length n sequence into lists of integers.

    key is the index passed to a call to __getitem__().
    """
    if type(key) is int:
        if -n <= key < 0:
            l = [key + n]
        elif 0 <= key < n:
            l = [key]
        else:
            raise IndexError('variable index out of range')
    elif type(key) is list and (not [k for k in key if type(k) is not int]) or (type(key) is matrix and key.typecode == 'i'):
        l = [k for k in key if -n <= k < n]
        if len(l) != len(key):
            raise IndexError('variable index out of range')
        for i in range(len(l)):
            if l[i] < 0:
                l[i] += n
    elif type(key) is slice:
        ind = key.indices(n)
        l = list(range(ind[0], ind[1], ind[2]))
    else:
        raise TypeError('invalid key')
    return l