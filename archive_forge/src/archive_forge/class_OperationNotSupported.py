from sympy.utilities import public
@public
class OperationNotSupported(BasePolynomialError):

    def __init__(self, poly, func):
        self.poly = poly
        self.func = func

    def __str__(self):
        return '`%s` operation not supported by %s representation' % (self.func, self.poly.rep.__class__.__name__)