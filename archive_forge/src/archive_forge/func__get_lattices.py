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
def _get_lattices(self, target_lattice, s, supercell_size=1):
    """
        Yields lattices for s with lengths and angles close to the lattice of target_s. If
        supercell_size is specified, the returned lattice will have that number of primitive
        cells in it.

        Args:
            target_lattice (Lattice): target lattice.
            s (Structure): input structure.
            supercell_size (int): Number of primitive cells in returned lattice
        """
    lattices = s.lattice.find_all_mappings(target_lattice, ltol=self.ltol, atol=self.angle_tol, skip_rotation_matrix=True)
    for latt, _, scale_m in lattices:
        if abs(abs(np.linalg.det(scale_m)) - supercell_size) < 0.5:
            yield (latt, scale_m)