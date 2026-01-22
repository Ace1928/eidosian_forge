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
class OrderDisorderElementComparator(AbstractComparator):
    """
    A Comparator that matches sites, given some overlap in the element
    composition.
    """

    def are_equal(self, sp1, sp2) -> bool:
        """
        True if there is some overlap in composition between the species.

        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            True always
        """
        set1 = set(sp1.elements)
        set2 = set(sp2.elements)
        return set1.issubset(set2) or set2.issubset(set1)

    def get_hash(self, composition):
        """Returns: Fractional composition."""
        return composition.fractional_composition