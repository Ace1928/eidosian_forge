from sympy.utilities import public
@public
class PolificationFailed(PolynomialError):

    def __init__(self, opt, origs, exprs, seq=False):
        if not seq:
            self.orig = origs
            self.expr = exprs
            self.origs = [origs]
            self.exprs = [exprs]
        else:
            self.origs = origs
            self.exprs = exprs
        self.opt = opt
        self.seq = seq

    def __str__(self):
        if not self.seq:
            return 'Cannot construct a polynomial from %s' % str(self.orig)
        else:
            return 'Cannot construct polynomials from %s' % ', '.join(map(str, self.origs))