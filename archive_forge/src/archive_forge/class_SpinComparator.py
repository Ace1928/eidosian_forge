from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors
class SpinComparator(AbstractComparator):
    """
    A Comparator that matches magnetic structures to their inverse spins.
    This comparator is primarily used to filter magnetically ordered
    structures with opposite spins, which are equivalent.
    """

    def are_equal(self, sp1, sp2) -> bool:
        """
        True if species are exactly the same, i.e., Fe2+ == Fe2+ but not
        Fe3+. and the spins are reversed. i.e., spin up maps to spin down,
        and vice versa.

        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            Boolean indicating whether species are equal.
        """
        for s1 in sp1:
            spin1 = getattr(s1, 'spin', 0) or 0
            oxi1 = getattr(s1, 'oxi_state', 0)
            for s2 in sp2:
                spin2 = getattr(s2, 'spin', 0) or 0
                oxi2 = getattr(s2, 'oxi_state', 0)
                if s1.symbol == s2.symbol and oxi1 == oxi2 and (spin2 == -spin1):
                    break
            else:
                return False
        return True

    def get_hash(self, composition):
        """Returns: Fractional composition."""
        return composition.fractional_composition