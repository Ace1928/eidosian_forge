from sympy.utilities import public
@public
class ComputationFailed(BasePolynomialError):

    def __init__(self, func, nargs, exc):
        self.func = func
        self.nargs = nargs
        self.exc = exc

    def __str__(self):
        return '%s(%s) failed without generators' % (self.func, ', '.join(map(str, self.exc.exprs[:self.nargs])))