from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

        Find a transitive subgroup of S6.

        Parameters
        ==========

        existing_gens : list of Permutation
            Optionally empty list of generators that must be in the group.

        needed_gen_orders : list of positive int
            Nonempty list of the orders of the additional generators that are
            to be found.

        order: int
            The order of the group being sought.

        alt: bool, None
            If True, require the group to be contained in A6.
            If False, require the group not to be contained in A6.

        profile : dict
            If given, the group's order profile must equal this.

        anti_profile : dict
            If given, the group's order profile must *not* equal this.

        