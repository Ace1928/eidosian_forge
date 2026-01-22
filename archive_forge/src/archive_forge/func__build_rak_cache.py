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
def _build_rak_cache(self) -> None:
    """Create explicit entries for for atoms so don't miss altlocs.

        This ensures that self.akc (atom key cache) has an entry for selected
        atom name (e.g. "CA") amongst any that have altlocs.  Without this,
        rak() on the other altloc atom first may result in the main atom being
        missed.
        """
    for ak in sorted(self.ak_set):
        atmName = ak.akl[3]
        if self.akc.get(atmName) is None:
            self.akc[atmName] = ak