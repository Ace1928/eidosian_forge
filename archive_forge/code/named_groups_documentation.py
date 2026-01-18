from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
Return a group of Rubik's cube generators

    >>> from sympy.combinatorics.named_groups import RubikGroup
    >>> RubikGroup(2).is_group
    True
    