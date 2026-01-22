import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _ln_exp_bound(self):
    """Compute a lower bound for the adjusted exponent of self.ln().
        In other words, compute r such that self.ln() >= 10**r.  Assumes
        that self is finite and positive and that self != 1.
        """
    adj = self._exp + len(self._int) - 1
    if adj >= 1:
        return len(str(adj * 23 // 10)) - 1
    if adj <= -2:
        return len(str((-1 - adj) * 23 // 10)) - 1
    op = _WorkRep(self)
    c, e = (op.int, op.exp)
    if adj == 0:
        num = str(c - 10 ** (-e))
        den = str(c)
        return len(num) - len(den) - (num < den)
    return e + len(str(10 ** (-e) - c)) - 1