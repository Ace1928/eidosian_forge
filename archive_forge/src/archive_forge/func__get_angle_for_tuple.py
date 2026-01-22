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
def _get_angle_for_tuple(self, angle_key: EKT) -> Optional[Union['Hedron', 'Dihedron']]:
    len_mkey = len(angle_key)
    rval: Optional[Union['Hedron', 'Dihedron']]
    if 4 == len_mkey:
        rval = self.dihedra.get(cast(DKT, angle_key), None)
    elif 3 == len_mkey:
        rval = self.hedra.get(cast(HKT, angle_key), None)
    else:
        return None
    return rval