from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
class VElement:
    """
    An element of some V_{0, n} which is the free R-module on P_{0, n}

    sage: m = PerfectMatching([(0, 1), (3, 4), (2, 5)])
    sage: v1 = VElement(m)
    sage: v1
    (1)*[(0, 1), (2, 5), (3, 4)]
    sage: v2 = (q + q**-1)*v1
    sage: v2
    (q^-1 + q)*[(0, 1), (2, 5), (3, 4)]
    sage: v3 = q* VElement(PerfectMatching([(5, 0), (4, 3), (1, 2)]))
    sage: v1 + v2 + v3
    (q^-1 + 1 + q)*[(0, 1), (2, 5), (3, 4)] + (q)*[(0, 5), (1, 2), (3, 4)]
    sage: v2.insert_cup(6)
    (q^-1 + q)*[(0, 1), (2, 5), (3, 4), (6, 7)]
    sage: (v1 + v2 + v3).cap_off(1)
    (q^-1 + 2 + q + q^2)*[(0, 3), (1, 2)]
    """

    def __init__(self, spec=None):
        self.dict = {}
        if spec is None:
            spec = PerfectMatching([])
        if isinstance(spec, dict):
            self.dict = spec
        if isinstance(spec, PerfectMatching):
            assert spec.is_noncrossing()
            self.dict[spec] = R.one()

    def __rmul__(self, other):
        if other in R:
            other = R(other)
            return VElement({m: other * c for m, c in self.dict.items()})

    def __add__(self, other):
        if isinstance(other, VElement):
            ans_dict = self.dict.copy()
            for matching, coeff in other.dict.items():
                cur_coeff = self.dict.get(matching, R.zero())
                ans_dict[matching] = cur_coeff + coeff
        return VElement(ans_dict)

    def __repr__(self):
        matchings = sorted(self.dict)
        terms = ['(%s)*%s' % (self.dict[m], m) for m in matchings]
        if len(terms) == 0:
            return '0'
        return ' + '.join(terms)

    def insert_cup(self, i):
        """
        Insert an new matching at (i, i + 1)
        """
        return VElement({insert_cup(m, i): c for m, c in self.dict.items()})

    def cap_off(self, i):
        ans_dict = {}
        for matching, coeff in self.dict.items():
            new_matching, has_circle = cap_off(matching, i)
            cur_coeff = ans_dict.get(new_matching, R.zero())
            if has_circle:
                coeff = (q + q ** (-1)) * coeff
            ans_dict[new_matching] = cur_coeff + coeff
        return VElement(ans_dict)

    def cap_then_cup(self, i):
        return self.cap_off(i).insert_cup(i)

    def add_positive_crossing(self, i):
        return self + -q * self.cap_then_cup(i)

    def add_negative_crossing(self, i):
        return self.cap_then_cup(i) + -q * self

    def is_multiple_of_empty_pairing(self):
        return len(self.dict) == 1 and PerfectMatching([]) in self.dict