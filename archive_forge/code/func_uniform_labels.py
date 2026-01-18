from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
def uniform_labels(self, mol1, mol2):
    """
        Args:
            mol1 (Molecule): Molecule 1
            mol2 (Molecule): Molecule 2.

        Returns:
            Labels
        """
    ob_mol1 = BabelMolAdaptor(mol1).openbabel_mol
    ob_mol2 = BabelMolAdaptor(mol2).openbabel_mol
    ilabel1, iequal_atom1, inchi1 = self._inchi_labels(ob_mol1)
    ilabel2, iequal_atom2, inchi2 = self._inchi_labels(ob_mol2)
    if inchi1 != inchi2:
        return (None, None)
    if iequal_atom1 != iequal_atom2:
        raise RuntimeError('Design Error! Equivalent atoms are inconsistent')
    vmol1 = self._virtual_molecule(ob_mol1, ilabel1, iequal_atom1)
    vmol2 = self._virtual_molecule(ob_mol2, ilabel2, iequal_atom2)
    if vmol1.NumAtoms() != vmol2.NumAtoms():
        return (None, None)
    if vmol1.NumAtoms() < 3 or self._is_molecule_linear(vmol1) or self._is_molecule_linear(vmol2):
        c_label1, c_label2 = self._assistant_mapper.uniform_labels(mol1, mol2)
    else:
        heavy_atom_indices2 = self._align_heavy_atoms(ob_mol1, ob_mol2, vmol1, vmol2, ilabel1, ilabel2, iequal_atom1)
        c_label1, c_label2 = self._align_hydrogen_atoms(ob_mol1, ob_mol2, ilabel1, heavy_atom_indices2)
    if c_label1 and c_label2:
        elements1 = self._get_elements(ob_mol1, c_label1)
        elements2 = self._get_elements(ob_mol2, c_label2)
        if elements1 != elements2:
            return (None, None)
    return (c_label1, c_label2)