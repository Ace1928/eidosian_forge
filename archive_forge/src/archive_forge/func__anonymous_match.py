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
def _anonymous_match(self, struct1: Structure, struct2: Structure, fu: int, s1_supercell=True, use_rms=False, break_on_match=False, single_match=False):
    """
        Tries all permutations of matching struct1 to struct2.

        Args:
            struct1 (Structure): First structure
            struct2 (Structure): Second structure
            fu (int): Factor of unit cell of struct1 to match to struct2
            s1_supercell (bool): whether to create the supercell of struct1 (vs struct2)
            use_rms (bool): Whether to minimize the rms of the matching
            break_on_match (bool): Whether to break search on first match
            single_match (bool): Whether to return only the best match

        Returns:
            List of (mapping, match)
        """
    if not isinstance(self._comparator, SpeciesComparator):
        raise ValueError('Anonymous fitting currently requires SpeciesComparator')
    sp1 = struct1.elements
    sp2 = struct2.elements
    if len(sp1) != len(sp2):
        return None
    ratio = fu if s1_supercell else 1 / fu
    swapped = len(struct1) * ratio < len(struct2)
    s1_comp = struct1.composition
    s2_comp = struct2.composition
    matches = []
    for perm in itertools.permutations(sp2):
        sp_mapping = dict(zip(sp1, perm))
        mapped_comp = Composition({sp_mapping[k]: v for k, v in s1_comp.items()})
        if not self._subset and self._comparator.get_hash(mapped_comp) != self._comparator.get_hash(s2_comp):
            continue
        mapped_struct = struct1.copy()
        mapped_struct.replace_species(sp_mapping)
        if swapped:
            match = self._strict_match(struct2, mapped_struct, fu, not s1_supercell, use_rms, break_on_match)
        else:
            match = self._strict_match(mapped_struct, struct2, fu, s1_supercell, use_rms, break_on_match)
        if match:
            matches.append((sp_mapping, match))
            if single_match:
                break
    return matches