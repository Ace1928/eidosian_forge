from ..libmp.backend import xrange
from .calculus import defun
def wrapsafe(f):

    def g(*args):
        try:
            return f(*args)
        except (ArithmeticError, ValueError):
            return 0
    return g