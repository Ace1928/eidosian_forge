from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
def to_structure_with_site_properties_Ucif(self) -> Structure:
    """Transfers this object into a structure with site properties (Ucif).
        This is useful for sorting the atoms in the structure including site properties.
        E.g., with code like this:
        def sort_order(site):
            return [site.specie.X, site.frac_coords[0], site.frac_coords[1], site.frac_coords[2]]
        new_structure0 = Structure.from_sites(sorted(structure0, key=sort_order)).

        Returns:
            Structure
        """
    site_properties: dict = {'U11_cif': [], 'U22_cif': [], 'U33_cif': [], 'U23_cif': [], 'U13_cif': [], 'U12_cif': []}
    if self.thermal_displacement_matrix_cif is None:
        cif_matrix = self.get_reduced_matrix(self.Ucif)
    else:
        cif_matrix = self.thermal_displacement_matrix_cif
    for atom_ucif in cif_matrix:
        site_properties['U11_cif'].append(atom_ucif[0])
        site_properties['U22_cif'].append(atom_ucif[1])
        site_properties['U33_cif'].append(atom_ucif[2])
        site_properties['U23_cif'].append(atom_ucif[3])
        site_properties['U13_cif'].append(atom_ucif[4])
        site_properties['U12_cif'].append(atom_ucif[5])
    return self.structure.copy(site_properties=site_properties)