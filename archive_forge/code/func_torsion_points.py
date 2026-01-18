from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence
def torsion_points(self):
    """
        Return torsion points of curve over Rational number.

        Return point objects those are finite order.
        According to Nagell-Lutz theorem, torsion point p(x, y)
        x and y are integers, either y = 0 or y**2 is divisor
        of discriminent. According to Mazur's theorem, there are
        at most 15 points in torsion collection.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(-43, 166)
        >>> sorted(e2.torsion_points())
        [(-5, -16), (-5, 16), O, (3, -8), (3, 8), (11, -32), (11, 32)]

        """
    if self.characteristic > 0:
        raise ValueError('No torsion point for Finite Field.')
    l = [EllipticCurvePoint.point_at_infinity(self)]
    for xx in solve(self._eq.subs({self.y: 0, self.z: 1})):
        if xx.is_rational:
            l.append(self(xx, 0))
    for i in divisors(self.discriminant, generator=True):
        j = int(i ** 0.5)
        if j ** 2 == i:
            for xx in solve(self._eq.subs({self.y: j, self.z: 1})):
                if not xx.is_rational:
                    continue
                p = self(xx, j)
                if p.order() != oo:
                    l.extend([p, -p])
    return l