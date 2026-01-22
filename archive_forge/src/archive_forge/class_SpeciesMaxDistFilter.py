from __future__ import annotations
import abc
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class SpeciesMaxDistFilter(AbstractStructureFilter):
    """This filter removes structures that do have two particular species that are
    not nearest neighbors by a predefined max_dist. For instance, if you are
    analyzing Li battery materials, you would expect that each Li+ would be
    nearest neighbor to lower oxidation state transition metal for
    electrostatic reasons. This only works if the structure is oxidation state
    decorated, as structures with only elemental sites are automatically
    assumed to have net charge of 0.
    """

    def __init__(self, sp1, sp2, max_dist):
        """
        Args:
            sp1 (Species): First specie
            sp2 (Species): Second specie
            max_dist (float): Maximum distance between species.
        """
        self.sp1 = get_el_sp(sp1)
        self.sp2 = get_el_sp(sp2)
        self.max_dist = max_dist

    def test(self, structure: Structure):
        """Method to execute the test.

        Args:
            structure (Structure): Input structure to test

        Returns:
            bool: True if structure does not contain the two species are distances
                greater than max_dist.
        """
        sp1_indices = [idx for idx, site in enumerate(structure) if site.specie == self.sp1]
        sp2_indices = [idx for idx, site in enumerate(structure) if site.specie == self.sp2]
        frac_coords1 = structure.frac_coords[sp1_indices, :]
        frac_coords2 = structure.frac_coords[sp2_indices, :]
        lattice = structure.lattice
        dists = lattice.get_all_distances(frac_coords1, frac_coords2)
        return all((any(row) for row in dists < self.max_dist))