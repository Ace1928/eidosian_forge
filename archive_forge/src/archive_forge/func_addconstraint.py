from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def addconstraint(self, c):
    """ 
        Adds constraint c to the list of constraints. 
        """
    if type(c) is not constraint:
        raise TypeError('argument must be of type constraint')
    if c.type() == '<':
        self._inequalities += [c]
    if c.type() == '=':
        self._equalities += [c]
    for v in c.variables():
        if c.type() == '<':
            if v in self._variables:
                self._variables[v]['i'] += [c]
            else:
                self._variables[v] = {'o': False, 'i': [c], 'e': []}
        elif v in self._variables:
            self._variables[v]['e'] += [c]
        else:
            self._variables[v] = {'o': False, 'i': [], 'e': [c]}