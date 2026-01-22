import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def _invs(self):
    """
        Return a dictionary with `{gen: inverse}` where `gen` is a rewriting
        generator of `codomain` (e.g. strong generator for permutation groups)
        and `inverse` is an element of its preimage

        """
    image = self.image()
    inverses = {}
    for k in list(self.images.keys()):
        v = self.images[k]
        if not (v in inverses or v.is_identity):
            inverses[v] = k
    if isinstance(self.codomain, PermutationGroup):
        gens = image.strong_gens
    else:
        gens = image.generators
    for g in gens:
        if g in inverses or g.is_identity:
            continue
        w = self.domain.identity
        if isinstance(self.codomain, PermutationGroup):
            parts = image._strong_gens_slp[g][::-1]
        else:
            parts = g
        for s in parts:
            if s in inverses:
                w = w * inverses[s]
            else:
                w = w * inverses[s ** (-1)] ** (-1)
        inverses[g] = w
    return inverses