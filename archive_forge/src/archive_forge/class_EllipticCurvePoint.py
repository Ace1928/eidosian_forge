from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence
class EllipticCurvePoint:
    """
    Point of Elliptic Curve

    Examples
    ========

    >>> from sympy.ntheory.elliptic_curve import EllipticCurve
    >>> e1 = EllipticCurve(-17, 16)
    >>> p1 = e1(0, -4, 1)
    >>> p2 = e1(1, 0)
    >>> p1 + p2
    (15, -56)
    >>> e3 = EllipticCurve(-1, 9)
    >>> e3(1, -3) * 3
    (664/169, 17811/2197)
    >>> (e3(1, -3) * 3).order()
    oo
    >>> e2 = EllipticCurve(-2, 0, 0, 1, 1)
    >>> p = e2(-1,1)
    >>> q = e2(0, -1)
    >>> p+q
    (4, 8)
    >>> p-q
    (1, 0)
    >>> 3*p-5*q
    (328/361, -2800/6859)
    """

    @staticmethod
    def point_at_infinity(curve):
        return EllipticCurvePoint(0, 1, 0, curve)

    def __init__(self, x, y, z, curve):
        dom = curve._domain.convert
        self.x = dom(x)
        self.y = dom(y)
        self.z = dom(z)
        self._curve = curve
        self._domain = self._curve._domain
        if not self._curve.__contains__(self):
            raise ValueError('The curve does not contain this point')

    def __add__(self, p):
        if self.z == 0:
            return p
        if p.z == 0:
            return self
        x1, y1 = (self.x / self.z, self.y / self.z)
        x2, y2 = (p.x / p.z, p.y / p.z)
        a1 = self._curve._a1
        a2 = self._curve._a2
        a3 = self._curve._a3
        a4 = self._curve._a4
        a6 = self._curve._a6
        if x1 != x2:
            slope = (y1 - y2) / (x1 - x2)
            yint = (y1 * x2 - y2 * x1) / (x2 - x1)
        else:
            if y1 + y2 == 0:
                return self.point_at_infinity(self._curve)
            slope = (3 * x1 ** 2 + 2 * a2 * x1 + a4 - a1 * y1) / (a1 * x1 + a3 + 2 * y1)
            yint = (-x1 ** 3 + a4 * x1 + 2 * a6 - a3 * y1) / (a1 * x1 + a3 + 2 * y1)
        x3 = slope ** 2 + a1 * slope - a2 - x1 - x2
        y3 = -(slope + a1) * x3 - yint - a3
        return self._curve(x3, y3, 1)

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __mul__(self, n):
        n = as_int(n)
        r = self.point_at_infinity(self._curve)
        if n == 0:
            return r
        if n < 0:
            return -self * -n
        p = self
        while n:
            if n & 1:
                r = r + p
            n >>= 1
            p = p + p
        return r

    def __rmul__(self, n):
        return self * n

    def __neg__(self):
        return EllipticCurvePoint(self.x, -self.y - self._curve._a1 * self.x - self._curve._a3, self.z, self._curve)

    def __repr__(self):
        if self.z == 0:
            return 'O'
        dom = self._curve._domain
        try:
            return '({}, {})'.format(dom.to_sympy(self.x), dom.to_sympy(self.y))
        except TypeError:
            pass
        return '({}, {})'.format(self.x, self.y)

    def __sub__(self, other):
        return self.__add__(-other)

    def order(self):
        """
        Return point order n where nP = 0.

        """
        if self.z == 0:
            return 1
        if self.y == 0:
            return 2
        p = self * 2
        if p.y == -self.y:
            return 3
        i = 2
        if self._domain != QQ:
            while int(p.x) == p.x and int(p.y) == p.y:
                p = self + p
                i += 1
                if p.z == 0:
                    return i
            return oo
        while p.x.numerator == p.x and p.y.numerator == p.y:
            p = self + p
            i += 1
            if i > 12:
                return oo
            if p.z == 0:
                return i
        return oo