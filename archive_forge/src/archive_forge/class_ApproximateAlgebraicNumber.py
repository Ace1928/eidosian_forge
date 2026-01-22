from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
class ApproximateAlgebraicNumber:
    """
    An algebraic number which we can compute
    to arbitrary precision.  Specified by a function
    where f(prec) is the number to 2^prec bits.
    """

    def __init__(self, defining_function):
        if defining_function in QQ:
            x = QQ(defining_function)
            defining_function = lambda prec: ComplexField(prec)(x)
        self.f = defining_function
        self._min_poly = None

    def __repr__(self):
        return '<ApproxAN: %s>' % CDF(self(100))

    @cached_method
    def __call__(self, prec):
        return self.f(prec)

    def min_polynomial(self, prec=100, degree=10):
        if self._min_poly is None:
            self_prec = self(prec)
            p = best_algdep_factor(self_prec, degree)
            z = self(2 * prec)
            if acceptable_error(p, z, ZZ(0), 0.2):
                self._min_poly = p
                self._default_precision = prec
                self._approx_root = self_prec
        return self._min_poly

    def express(self, a, prec=None):
        """Express the given number in terms of self"""
        if self._min_poly is None:
            raise ValueError('Minimal polynomial is not known.')
        if prec is None:
            prec = self._default_precision
        p = self._min_poly
        z0, a0 = (self(prec), a(prec))
        A = complex_to_lattice(z0, p.degree(), a0)
        v = A.LLL(delta=0.75)[0]
        v = list(v)[:-2]
        if v[-1] == 0:
            return None
        R = PolynomialRing(QQ, 'x')
        q = -R(v[:-1]) / v[-1]
        z1, a1 = (self(2 * prec), a(2 * prec))
        if acceptable_error(q, z1, a1, 0.2):
            return q

    def express_several(self, elts, prec=None):
        """
        Return exact expressions every number elts, provided this is
        possible.
        """
        ans = []
        for a in elts:
            exact = self.express(a, prec)
            if exact is None:
                return None
            else:
                ans.append(exact)
        return ans

    def can_express(self, a, prec=None):
        return self.express(a, prec) is not None

    def number_field(self):
        p = self._min_poly
        if p is None:
            raise ValueError('Minimal polynomial is not known.')
        q = p.change_ring(QQ)
        q = 1 / q.leading_coefficient() * q
        return NumberField(q, 'z', embedding=self._approx_root)

    def place(self, prec):
        K = self.number_field()
        z = self(prec)
        CC = z.parent()
        return K.hom(z, check=False, codomain=CC)

    def __add__(self, other):
        if not isinstance(other, ApproximateAlgebraicNumber):
            raise ValueError

        def f(prec):
            return self(prec) + other(prec)
        return ApproximateAlgebraicNumber(f)

    def __mul__(self, other):
        if not isinstance(other, ApproximateAlgebraicNumber):
            raise ValueError

        def f(prec):
            return self(prec) * other(prec)
        return ApproximateAlgebraicNumber(f)

    def __div__(self, other):
        if not isinstance(other, ApproximateAlgebraicNumber):
            raise ValueError

        def f(prec):
            return self(prec) / other(prec)
        return ApproximateAlgebraicNumber(f)

    def __pow__(self, n):

        def f(prec):
            return self(prec) ** n
        return ApproximateAlgebraicNumber(f)

    def __neg__(self):

        def f(prec):
            return -self(prec)
        return ApproximateAlgebraicNumber(f)

    def __sub__(self, other):
        return self + other.__neg__()