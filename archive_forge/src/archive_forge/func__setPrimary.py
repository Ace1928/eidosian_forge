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
def _setPrimary(self) -> bool:
    """Mark dihedra required for psi, phi, omega, chi and other angles."""
    dhc = self.e_class
    if dhc == 'NCACN':
        self.pclass = self.re_class[0:7] + 'XN'
        self.primary = True
    elif dhc == 'CACNCA':
        self.pclass = 'XCAXC' + self.re_class[5:]
        self.primary = True
    elif dhc == 'CNCAC':
        self.pclass = 'XC' + self.re_class[2:]
        self.primary = True
    elif dhc == 'CNCACB':
        self.altCB_class = 'XC' + self.re_class[2:]
        self.primary = False
    elif dhc in primary_angles:
        self.primary = True
        self.pclass = self.re_class
    else:
        self.primary = False