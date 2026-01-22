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
class SpeciesComparator(AbstractComparator):
    """A Comparator that matches species exactly. The default used in StructureMatcher."""

    def are_equal(self, sp1, sp2) -> bool:
        """
        True if species are exactly the same, i.e., Fe2+ == Fe2+ but not Fe3+.

        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            Boolean indicating whether species are equal.
        """
        return sp1 == sp2

    def get_hash(self, composition: Composition):
        """Returns: Fractional composition."""
        return composition.fractional_composition