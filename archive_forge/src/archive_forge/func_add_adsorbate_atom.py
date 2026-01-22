from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def add_adsorbate_atom(self, indices: list[int], species: str | Element | Species, distance: float, specie: Species | Element | str | None=None) -> Self:
    """Add adsorbate onto the Slab, along the c lattice vector.

        Args:
            indices (list[int]): Indices of sites on which to put the adsorbate.
                Adsorbate will be placed relative to the center of these sites.
            species (str | Element | Species): The species to add.
            distance (float): between centers of the adsorbed atom and the
                given site in Angstroms, along the c lattice vector.
            specie: Deprecated argument in #3691. Use 'species' instead.

        Returns:
            Slab: self with adsorbed atom.
        """
    if specie is not None:
        warnings.warn("The argument 'specie' is deprecated. Use 'species' instead.", DeprecationWarning)
        species = specie
    center = np.sum([self[idx].coords for idx in indices], axis=0) / len(indices)
    coords = center + self.normal * distance
    self.append(species, coords, coords_are_cartesian=True)
    return self