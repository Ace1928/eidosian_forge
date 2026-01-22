import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
class _FactorizedSqrtLinCombination:

    def __init__(self, d={}, embed_cache=None):
        self._dict = _filter_zero(d)
        self._embed_cache = embed_cache

    def _real_mpfi_(self, RIF):

        def eval_term(k, v):
            pr = prod([_to_RIF(t, RIF, self._embed_cache) for t in k], RIF(1))
            if not pr > 0:
                raise _SqrtException()
            return pr.sqrt() * _to_RIF(v, RIF, self._embed_cache)
        return sum([eval_term(k, v) for k, v in self._dict.items()], RIF(0))

    def __repr__(self):
        if not self._dict:
            return '0'

        def term(item):
            k, v = item
            b = '(%r)' % v
            for s in k:
                b += ' * sqrt(%r)' % s
            return b
        return '+'.join([term(item) for item in self._dict.items()])

    @staticmethod
    def from_sqrt_lin_combination(l):
        """
        Construct from a SqrtLinCombination.
        """

        def to_set(k):
            if k == _One:
                return frozenset()
            else:
                return frozenset([k])
        return _FactorizedSqrtLinCombination(dict(((to_set(k), v) for k, v in l._dict.items())), embed_cache=l._embed_cache)

    def __add__(self, other):
        d = {}
        for k, v in self._dict.items():
            d[k] = d.get(k, 0) + v
        for k, v in other._dict.items():
            d[k] = d.get(k, 0) + v
        return _FactorizedSqrtLinCombination(d, embed_cache=_get_embed_cache(self, other))

    def __neg__(self):
        return _FactorizedSqrtLinCombination(dict(((k, -v) for k, v in self._dict.items())), embed_cache=self._embed_cache)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        d = {}
        for k1, v1 in self._dict.items():
            for k2, v2 in other._dict.items():
                k = k1 ^ k2
                v = v1 * v2 * prod(k1 & k2, _One)
                d[k] = d.get(k, 0) + v
        return _FactorizedSqrtLinCombination(d, embed_cache=_get_embed_cache(self, other))

    def is_zero(self):
        """
        Returns True if it is zero, False otherwise.
        """
        if not self._dict:
            return True
        if len(self._dict) == 1:
            return _first(self._dict.values()) == 0
        common_terms = reduce(operator.and_, self._dict.keys())
        d = dict(((k - common_terms, v) for k, v in self._dict.items()))
        term = _firstfirst(d.keys())
        left = _FactorizedSqrtLinCombination(dict(((k, v) for k, v in d.items() if term in k)), embed_cache=self._embed_cache)
        right = _FactorizedSqrtLinCombination(dict(((k, v) for k, v in d.items() if term not in k)), embed_cache=self._embed_cache)
        if not (left * left - right * right).is_zero():
            return False
        if left.is_zero():
            return True
        prec = 53
        while True:
            opposite_signs = _opposite_signs(left, right, prec)
            if opposite_signs is not None:
                return opposite_signs
            prec *= 2