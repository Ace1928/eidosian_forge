import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
def update_dCoordSpace(self, workSelector: Optional[np.ndarray]=None) -> None:
    """Compute/update coordinate space transforms for chain dihedra.

        Requires all atoms updated so calls :meth:`.assemble_residues`
        (returns immediately if all atoms already assembled).

        :param [bool] workSelector:
            Optional mask to select dihedra for update
        """
    if workSelector is None:
        self.assemble_residues()
        workSelector = np.logical_not(self.dcsValid)
    workSet = self.dSet[workSelector]
    self.dCoordSpace[:, workSelector] = multi_coord_space(workSet, np.sum(workSelector), True)
    self.dcsValid[workSelector] = True