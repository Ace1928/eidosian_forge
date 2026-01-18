from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
from sympy.utilities.iterables import uniq

    Returns the direct product of several groups as a permutation group.

    Explanation
    ===========

    This is implemented much like the __mul__ procedure for taking the direct
    product of two permutation groups, but the idea of shifting the
    generators is realized in the case of an arbitrary number of groups.
    A call to DirectProduct(G1, G2, ..., Gn) is generally expected to be faster
    than a call to G1*G2*...*Gn (and thus the need for this algorithm).

    Examples
    ========

    >>> from sympy.combinatorics.group_constructs import DirectProduct
    >>> from sympy.combinatorics.named_groups import CyclicGroup
    >>> C = CyclicGroup(4)
    >>> G = DirectProduct(C, C, C)
    >>> G.order()
    64

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.__mul__

    