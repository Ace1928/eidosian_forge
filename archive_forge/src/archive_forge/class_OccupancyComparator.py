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
class OccupancyComparator(AbstractComparator):
    """
    A Comparator that matches occupancies on sites,
    irrespective of the species of those sites.
    """

    def are_equal(self, sp1, sp2) -> bool:
        """
        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            bool: True if sets of occupancies (amt) are equal on both sites.
        """
        return set(sp1.element_composition.values()) == set(sp2.element_composition.values())

    def get_hash(self, composition):
        """
        Args:
            composition: Composition.

        TODO: might need a proper hash method

        Returns:
            1. Difficult to define sensible hash
        """
        return 1