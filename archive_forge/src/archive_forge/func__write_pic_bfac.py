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
def _write_pic_bfac(self, atm: Atom, s: str, col: int) -> Tuple[str, int]:
    ak = self.rak(atm)
    if 0 == col % 5:
        s += 'BFAC:'
    s += ' ' + ak.id + ' ' + f'{atm.get_bfactor():6.2f}'
    col += 1
    if 0 == col % 5:
        s += '\n'
    return (s, col)