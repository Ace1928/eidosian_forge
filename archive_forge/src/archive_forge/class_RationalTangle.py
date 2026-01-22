import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
class RationalTangle(Tangle):
    """
    A rational tangle. ``RationalTangle(a, b)`` gives the a/b rational tangle when ``a``
    and ``b`` are integers. If ``q`` is a rational, then ``RationalTangle(q)`` gives the
    corresponding rational tangle.

    This is a class that extends Tangle since it provides some additional information as
    attributes: ``fraction`` gives (a, b) and ``partial_quotients`` gives the continued
    fraction expansion of ``abs(a)/b``.

    >>> RationalTangle(2,5).braid_closure().exterior().identify() # doctest: +SNAPPY
    [m004(0,0), 4_1(0,0), K2_1(0,0), K4a1(0,0), otet02_00001(0,0)]
    """

    def __init__(self, a, b=1):
        if b == 1 and hasattr(a, 'numerator') and hasattr(a, 'denominator') and (not isinstance(a, int)):
            a, b = (a.numerator(), a.denominator())
        if b < 0:
            a, b = (-a, -b)
        self.fraction = (a, b)
        self.partial_quotients = pqs = continued_fraction_expansion(abs(a), b)
        T = InfinityTangle()
        for p in reversed(pqs):
            T = IntegerTangle(p) + T.invert()
        if a < 0:
            T = -T
        Tangle.__init__(self, 2, T.crossings, T.adjacent, f'RationalTangle({a}, {b})')