from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
class Hyper_Function(Expr):
    """ A generalized hypergeometric function. """

    def __new__(cls, ap, bq):
        obj = super().__new__(cls)
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.ap, self.bq)

    @property
    def sizes(self):
        return (len(self.ap), len(self.bq))

    @property
    def gamma(self):
        """
        Number of upper parameters that are negative integers

        This is a transformation invariant.
        """
        return sum((bool(x.is_integer and x.is_negative) for x in self.ap))

    def _hashable_content(self):
        return super()._hashable_content() + (self.ap, self.bq)

    def __call__(self, arg):
        return hyper(self.ap, self.bq, arg)

    def build_invariants(self):
        """
        Compute the invariant vector.

        Explanation
        ===========

        The invariant vector is:
            (gamma, ((s1, n1), ..., (sk, nk)), ((t1, m1), ..., (tr, mr)))
        where gamma is the number of integer a < 0,
              s1 < ... < sk
              nl is the number of parameters a_i congruent to sl mod 1
              t1 < ... < tr
              ml is the number of parameters b_i congruent to tl mod 1

        If the index pair contains parameters, then this is not truly an
        invariant, since the parameters cannot be sorted uniquely mod1.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Hyper_Function
        >>> from sympy import S
        >>> ap = (S.Half, S.One/3, S(-1)/2, -2)
        >>> bq = (1, 2)

        Here gamma = 1,
             k = 3, s1 = 0, s2 = 1/3, s3 = 1/2
                    n1 = 1, n2 = 1,   n2 = 2
             r = 1, t1 = 0
                    m1 = 2:

        >>> Hyper_Function(ap, bq).build_invariants()
        (1, ((0, 1), (1/3, 1), (1/2, 2)), ((0, 2),))
        """
        abuckets, bbuckets = (sift(self.ap, _mod1), sift(self.bq, _mod1))

        def tr(bucket):
            bucket = list(bucket.items())
            if not any((isinstance(x[0], Mod) for x in bucket)):
                bucket.sort(key=lambda x: default_sort_key(x[0]))
            bucket = tuple([(mod, len(values)) for mod, values in bucket if values])
            return bucket
        return (self.gamma, tr(abuckets), tr(bbuckets))

    def difficulty(self, func):
        """ Estimate how many steps it takes to reach ``func`` from self.
            Return -1 if impossible. """
        if self.gamma != func.gamma:
            return -1
        oabuckets, obbuckets, abuckets, bbuckets = [sift(params, _mod1) for params in (self.ap, self.bq, func.ap, func.bq)]
        diff = 0
        for bucket, obucket in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
            for mod in set(list(bucket.keys()) + list(obucket.keys())):
                if mod not in bucket or mod not in obucket or len(bucket[mod]) != len(obucket[mod]):
                    return -1
                l1 = list(bucket[mod])
                l2 = list(obucket[mod])
                l1.sort()
                l2.sort()
                for i, j in zip(l1, l2):
                    diff += abs(i - j)
        return diff

    def _is_suitable_origin(self):
        """
        Decide if ``self`` is a suitable origin.

        Explanation
        ===========

        A function is a suitable origin iff:
        * none of the ai equals bj + n, with n a non-negative integer
        * none of the ai is zero
        * none of the bj is a non-positive integer

        Note that this gives meaningful results only when none of the indices
        are symbolic.

        """
        for a in self.ap:
            for b in self.bq:
                if (a - b).is_integer and (a - b).is_negative is False:
                    return False
        for a in self.ap:
            if a == 0:
                return False
        for b in self.bq:
            if b.is_integer and b.is_nonpositive:
                return False
        return True