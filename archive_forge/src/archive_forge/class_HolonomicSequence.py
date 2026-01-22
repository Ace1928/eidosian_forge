from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify
class HolonomicSequence:
    """
    A Holonomic Sequence is a type of sequence satisfying a linear homogeneous
    recurrence relation with Polynomial coefficients. Alternatively, A sequence
    is Holonomic if and only if its generating function is a Holonomic Function.
    """

    def __init__(self, recurrence, u0=[]):
        self.recurrence = recurrence
        if not isinstance(u0, list):
            self.u0 = [u0]
        else:
            self.u0 = u0
        if len(self.u0) == 0:
            self._have_init_cond = False
        else:
            self._have_init_cond = True
        self.n = recurrence.parent.base.gens[0]

    def __repr__(self):
        str_sol = 'HolonomicSequence(%s, %s)' % (self.recurrence.__repr__(), sstr(self.n))
        if not self._have_init_cond:
            return str_sol
        else:
            cond_str = ''
            seq_str = 0
            for i in self.u0:
                cond_str += ', u(%s) = %s' % (sstr(seq_str), sstr(i))
                seq_str += 1
            sol = str_sol + cond_str
            return sol
    __str__ = __repr__

    def __eq__(self, other):
        if self.recurrence == other.recurrence:
            if self.n == other.n:
                if self._have_init_cond and other._have_init_cond:
                    if self.u0 == other.u0:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False
        else:
            return False