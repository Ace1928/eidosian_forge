from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.util.due import Doi, due
def are_equal(self, mol1, mol2) -> bool:
    """
        Compare the bond table of the two molecules.

        Args:
            mol1: first molecule. pymatgen Molecule object.
            mol2: second molecules. pymatgen Molecule object.
        """
    b1 = set(self._get_bonds(mol1))
    b2 = set(self._get_bonds(mol2))
    return b1 == b2